"""GPU-resident JPEG images, resize/letterbox, and BEV rasterization."""
import base64
import ctypes as ct

import cupy as cp
import cv2
import numpy as np


class JpegImage(ct.Structure):
    _fields_ = [('channel', ct.c_void_p * 4), ('pitch', ct.c_size_t * 4)]


class NvJpeg:
    def __init__(self, decode_threads=1, prefer_hardware=True):
        self.lib = ct.CDLL('/usr/local/cuda/lib64/libnvjpeg.so')
        pointer, size, integer = ct.c_void_p, ct.c_size_t, ct.c_int
        signatures = {
            'nvjpegCreateSimple': [ct.POINTER(pointer)],
            'nvjpegCreate': [integer, pointer, ct.POINTER(pointer)],
            'nvjpegJpegStateCreate': [pointer, ct.POINTER(pointer)],
            'nvjpegGetImageInfo': [pointer, pointer, size, ct.POINTER(integer), ct.POINTER(integer), ct.POINTER(integer), ct.POINTER(integer)],
            'nvjpegDecodeBatchedInitialize': [pointer, pointer, integer, integer, integer],
            'nvjpegDecodeBatched': [pointer, pointer, ct.POINTER(pointer), ct.POINTER(size), ct.POINTER(JpegImage), pointer],
            'nvjpegEncoderStateCreate': [pointer, ct.POINTER(pointer), pointer],
            'nvjpegEncoderParamsCreate': [pointer, ct.POINTER(pointer), pointer],
            'nvjpegEncoderParamsSetQuality': [pointer, integer, pointer],
            'nvjpegEncoderParamsSetSamplingFactors': [pointer, integer, pointer],
            'nvjpegEncodeImage': [pointer, pointer, pointer, ct.POINTER(JpegImage), integer, integer, integer, pointer],
            'nvjpegEncodeRetrieveBitstream': [pointer, pointer, pointer, ct.POINTER(size), pointer],
            'nvjpegEncoderStateDestroy': [pointer], 'nvjpegEncoderParamsDestroy': [pointer],
            'nvjpegJpegStateDestroy': [pointer], 'nvjpegDestroy': [pointer],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name); function.argtypes = args; function.restype = integer
        self.handle, self.state, self.encoder, self.params = (pointer() for _ in range(4))
        # Thor exposes a JPEG decoder that the default hybrid backend does not select.
        hardware_status = self.lib.nvjpegCreate(3, None, ct.byref(self.handle)) if prefer_hardware else -1
        self.backend = 'nvJPEG hardware' if hardware_status == 0 else 'nvJPEG hybrid'
        self.fallback = None
        self.last_decode_backend = self.backend
        if hardware_status != 0:
            self.handle = pointer()
            self.check(self.lib.nvjpegCreateSimple(ct.byref(self.handle)))
        self.check(self.lib.nvjpegJpegStateCreate(self.handle, ct.byref(self.state)))
        self.check(self.lib.nvjpegDecodeBatchedInitialize(self.handle, self.state, 6, decode_threads, 6))
        stream = cp.cuda.get_current_stream().ptr
        self.check(self.lib.nvjpegEncoderStateCreate(self.handle, ct.byref(self.encoder), stream))
        self.check(self.lib.nvjpegEncoderParamsCreate(self.handle, ct.byref(self.params), stream))
        self.check(self.lib.nvjpegEncoderParamsSetSamplingFactors(self.params, 2, stream))
        self.encoders = [(self.encoder, self.params)]

    @staticmethod
    def check(code):
        if code: raise RuntimeError(f'nvJPEG error {code}')

    @staticmethod
    def descriptor(image):
        result = JpegImage(); result.channel[0] = image.data.ptr; result.pitch[0] = image.strides[0]
        return result

    def decode(self, encoded):
        if len(encoded) != 6: raise ValueError('Expected six JPEG camera images')
        pointers, lengths = (ct.c_void_p * 6)(), (ct.c_size_t * 6)()
        descriptors = (JpegImage * 6)(); images = []
        for i, data in enumerate(encoded):
            components, subsampling = ct.c_int(), ct.c_int()
            widths, heights = (ct.c_int * 4)(), (ct.c_int * 4)()
            self.check(self.lib.nvjpegGetImageInfo(self.handle, data.ctypes.data, data.nbytes,
                       ct.byref(components), ct.byref(subsampling), widths, heights))
            if not 0 < widths[0] <= 8192 or not 0 < heights[0] <= 8192:
                raise ValueError('Invalid JPEG image size')
            image = cp.empty((heights[0], widths[0], 3), dtype=cp.uint8)
            images.append(image); descriptors[i] = self.descriptor(image)
            pointers[i] = data.ctypes.data; lengths[i] = data.nbytes
        stream = cp.cuda.get_current_stream()
        code = self.lib.nvjpegDecodeBatched(self.handle, self.state, pointers, lengths, descriptors, stream.ptr)
        if code == 4 and self.backend == 'nvJPEG hardware':  # JPEG_NOT_SUPPORTED (e.g. progressive)
            stream.synchronize()
            if self.fallback is None:
                self.fallback = NvJpeg(prefer_hardware=False)
            self.last_decode_backend = self.fallback.backend
            return self.fallback.decode(encoded)
        self.check(code)
        self.last_decode_backend = self.backend
        # nvJPEG consumes host bitstreams asynchronously. Keep them alive until completion.
        stream.synchronize()
        return images

    def encode(self, image, quality=85):
        return self.encode_many([image], [quality])[0]

    def encode_many(self, images, qualities=None):
        """Submit all frame images before waiting for JPEG sizes and host copies."""
        images = [cp.ascontiguousarray(image, dtype=cp.uint8) for image in images]
        qualities = [85] * len(images) if qualities is None else qualities
        if len(qualities) != len(images):
            raise ValueError('One JPEG quality is required per image')
        stream = cp.cuda.get_current_stream()
        while len(self.encoders) < len(images):
            encoder, params = ct.c_void_p(), ct.c_void_p()
            self.check(self.lib.nvjpegEncoderStateCreate(self.handle, ct.byref(encoder), stream.ptr))
            self.check(self.lib.nvjpegEncoderParamsCreate(self.handle, ct.byref(params), stream.ptr))
            self.check(self.lib.nvjpegEncoderParamsSetSamplingFactors(params, 2, stream.ptr))
            self.encoders.append((encoder, params))
        lengths = [ct.c_size_t() for _ in images]
        descriptors = [self.descriptor(image) for image in images]
        for image, quality, descriptor, length, (encoder, params) in zip(
                images, qualities, descriptors, lengths, self.encoders):
            height, width = image.shape[:2]
            self.check(self.lib.nvjpegEncoderParamsSetQuality(params, quality, stream.ptr))
            self.check(self.lib.nvjpegEncodeImage(self.handle, encoder, params,
                       ct.byref(descriptor), 6, width, height, stream.ptr))
            self.check(self.lib.nvjpegEncodeRetrieveBitstream(self.handle, encoder, None,
                       ct.byref(length), stream.ptr))
        stream.synchronize()
        buffers = [np.empty(length.value, dtype=np.uint8) for length in lengths]
        for data, length, (encoder, _) in zip(buffers, lengths, self.encoders):
            self.check(self.lib.nvjpegEncodeRetrieveBitstream(self.handle, encoder,
                       data.ctypes.data, ct.byref(length), stream.ptr))
        stream.synchronize()
        return ['data:image/jpeg;base64,' + base64.b64encode(data[:length.value]).decode()
                for data, length in zip(buffers, lengths)]

    def close(self):
        if self.fallback is not None:
            self.fallback.close()
            self.fallback = None
        for encoder, params in self.encoders:
            for function, value in [('nvjpegEncoderParamsDestroy', params), ('nvjpegEncoderStateDestroy', encoder)]:
                if value: getattr(self.lib, function)(value); value.value = None
        self.encoders.clear()
        for function, value in [('nvjpegJpegStateDestroy',self.state),('nvjpegDestroy',self.handle)]:
            if value: getattr(self.lib,function)(value); value.value = None


_RESIZE = r'''
extern "C" __global__ void resize_bgr(const unsigned char* src, unsigned char* dst,
 int sw,int sh,int dw,int dh) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=dw*dh*3)return;
 int c=i%3, x=(i/3)%dw,y=i/(dw*3);
 float sx=fmaxf(0.f,fminf(sw-1.f,(x+.5f)*sw/dw-.5f));
 float sy=fmaxf(0.f,fminf(sh-1.f,(y+.5f)*sh/dh-.5f));
 int x0=(int)sx,y0=(int)sy,x1=min(x0+1,sw-1),y1=min(y0+1,sh-1);
 float fx=sx-x0,fy=sy-y0;
 float v=(1-fy)*((1-fx)*src[(y0*sw+x0)*3+c]+fx*src[(y0*sw+x1)*3+c])+
 fy*((1-fx)*src[(y1*sw+x0)*3+c]+fx*src[(y1*sw+x1)*3+c]);
 dst[i]=(unsigned char)fminf(255.f,fmaxf(0.f,roundf(v)));
}
extern "C" __global__ void letterbox(const unsigned char* src,float* dst,
 int sw,int sh,int size,int nw,int nh,int left,int top) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=size*size)return;
 int x=i%size,y=i/size;
 float rgb[3]={114.f,114.f,114.f};
 if(x>=left&&x<left+nw&&y>=top&&y<top+nh){
 float sx=fmaxf(0.f,fminf(sw-1.f,(x-left+.5f)*sw/nw-.5f));
 float sy=fmaxf(0.f,fminf(sh-1.f,(y-top+.5f)*sh/nh-.5f));
 int x0=(int)sx,y0=(int)sy,x1=min(x0+1,sw-1),y1=min(y0+1,sh-1);
 float fx=sx-x0,fy=sy-y0;
 for(int c=0;c<3;c++)rgb[2-c]=roundf(
 (1-fy)*((1-fx)*src[(y0*sw+x0)*3+c]+fx*src[(y0*sw+x1)*3+c])+
 fy*((1-fx)*src[(y1*sw+x0)*3+c]+fx*src[(y1*sw+x1)*3+c]));
 }
 for(int c=0;c<3;c++)dst[c*size*size+i]=rgb[c]/255.f;
}
'''
_MODULE = cp.RawModule(code=_RESIZE)


def resize(image, width, height):
    output = cp.empty((height,width,3), dtype=cp.uint8)
    _MODULE.get_function('resize_bgr')(((output.size+255)//256,), (256,),
        (image,output,np.int32(image.shape[1]),np.int32(image.shape[0]),np.int32(width),np.int32(height)))
    return output


def prepare(image, output):
    height,width = image.shape[:2]; size = output.shape[-1]
    ratio = min(size/height,size/width); nh,nw=round(height*ratio),round(width*ratio)
    top,left=(size-nh)//2,(size-nw)//2
    _MODULE.get_function('letterbox')(((size*size+255)//256,), (256,),
        (image,output,np.int32(width),np.int32(height),np.int32(size),np.int32(nw),np.int32(nh),np.int32(left),np.int32(top)))
    return height,width,ratio,top,left


class Rasterizer:
    def __init__(self, size=512):
        self.size=size
        # Static glyph masks and ego marker are uploaded once; frames stay on GPU.
        self.glyphs=np.zeros((128,10,7),dtype=np.uint8)
        for i in range(32,127): cv2.putText(self.glyphs[i],chr(i),(0,8),cv2.FONT_HERSHEY_SIMPLEX,.3,255,1,cv2.LINE_8)
        self.glyphs=cp.asarray(self.glyphs)
        marker=np.zeros((size,size),np.uint8)
        cv2.drawMarker(marker,(size//2,size//2),255,cv2.MARKER_TRIANGLE_UP,13,2)
        self.marker=cp.asarray(marker)>0

    def background(self,density,height):
        present=density>0; h=cp.clip(height/3,0,1); bright=cp.clip(cp.log1p(density)/2,.35,1)
        image=cp.empty((self.size,self.size,3),dtype=cp.uint8); image[:]=cp.asarray([22,19,13],dtype=cp.uint8)
        image[:,:,0]=cp.where(present,(100+120*(1-h))*bright,22)
        image[:,:,1]=cp.where(present,(100+140*h)*bright,19)
        image[:,:,2]=cp.where(present,(55+150*h)*bright,13)
        offsets=cp.arange(0,self.size,round(self.size/10)); image[offsets,:,:]=cp.asarray([40,38,30],dtype=cp.uint8); image[:,offsets,:]=cp.asarray([40,38,30],dtype=cp.uint8)
        image[self.marker]=cp.asarray([230,235,215],dtype=cp.uint8)
        return image

    def boxes(self,image,detections,colors,extent=50.0):
        if not detections:return image
        commands=[];labels=[]
        for d in detections:
            x,y,_=d['center'];sx,sy,_=d['size']
            a=int((extent-y-sy/2)*self.size/(2*extent));b=int((extent-x-sx/2)*self.size/(2*extent))
            c=int((extent-y+sy/2)*self.size/(2*extent));e=int((extent-x+sx/2)*self.size/(2*extent))
            commands.append([a,b,c,e,*colors[d['class_id']][::-1]])
            label=f"{d['label']} #{d['id']}".encode('ascii')[:24];labels.append(list(label)+[0]*(24-len(label)))
        commands=cp.asarray(commands,dtype=cp.int32);labels=cp.asarray(labels,dtype=cp.uint8)
        _DRAW((len(detections),),(256,),(image,commands,labels,self.glyphs,np.int32(self.size)))
        return image


_DRAW=cp.RawKernel(r'''
extern "C" __global__ void draw(unsigned char* img,const int* boxes,const unsigned char* text,
 const unsigned char* glyphs,int size){
 int d=blockIdx.x;const int* b=boxes+d*7;
 int width=max(0,b[2]-b[0]+1),height=max(0,b[3]-b[1]+1);
 for(int k=threadIdx.x;k<2*width+2*height;k+=blockDim.x){
 int x,y;
 if(k<width){x=b[0]+k;y=b[1];}else if(k<2*width){x=b[0]+k-width;y=b[3];}
 else if(k<2*width+height){x=b[0];y=b[1]+k-2*width;}else{x=b[2];y=b[1]+k-2*width-height;}
 if(x>=0&&x<size&&y>=0&&y<size)for(int c=0;c<3;c++)img[(y*size+x)*3+c]=b[4+c];
 }
 for(int k=threadIdx.x;k<24*10*7;k+=blockDim.x){
 int letter=k/70,p=k%70;unsigned char ch=text[d*24+letter];
 if(ch&&glyphs[ch*70+p]){int x=b[0]+letter*7+p%7,y=b[1]-12+p/7;
 if(x>=0&&x<size&&y>=0&&y<size)for(int c=0;c<3;c++)img[(y*size+x)*3+c]=b[4+c];}
 }
}
''','draw')
