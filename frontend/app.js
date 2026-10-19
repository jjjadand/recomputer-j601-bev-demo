import * as THREE from '/vendor/three.module.js';
import {t,className,bindTranslations,setLanguage,currentLanguage} from './i18n.js';

const $ = id => document.getElementById(id);
const colors = {0:'#ffbf52',1:'#d294ff',2:'#42ecb0',3:'#ee8ff3',5:'#6fb5ff',7:'#46b7ff',9:'#ffe164',11:'#ff6e6e'};
const channelNames = ['FRONT','FRONT RIGHT','FRONT LEFT','BACK','BACK LEFT','BACK RIGHT'];
let latest = null, displayedFusion = null, state = {}, telemetry = {}, view = 'compare', projection = true, frameVersion = 0;
let lastFrame = 0, wsOpen = false;
const rates = [], fpsSamples = [];
let monitor='data', sceneSignature='', portSignature='';
let loadLimit=-1;
let section='sensors';
try {
  const saved=localStorage.getItem('thor-section');
  if(['sensors','fusion','network'].includes(saved))section=saved;
} catch {}
bindTranslations();
setLanguage(currentLanguage());
const formatBytes = value => {
  if (!Number.isFinite(value)) return '—';
  if (value > 1e9) return `${(value/1e9).toFixed(2)} GB`;
  if (value > 1e6) return `${(value/1e6).toFixed(1)} MB`;
  return `${(value/1e3).toFixed(1)} KB`;
};
const timeLabel = seconds => `${String(Math.floor(seconds/60)).padStart(2,'0')}:${(seconds%60).toFixed(1).padStart(4,'0')}`;
const frameTime = index => state.frame_times?.[index] ?? index/(state.native_fps || 2);
const rateLabel = value => value>=1000?`${(value/1000).toFixed(2)} Gbps`:`${value.toFixed(1)} Mbps`;
const rateHTML = value => value>=1000?`${(value/1000).toFixed(2)}<small>Gbps</small>`:`${value.toFixed(1)}<small>Mbps</small>`;

for (let i=0;i<6;i++) {
  const card = document.createElement('div'); card.className='camera-card';
  card.innerHTML=`<canvas width="640" height="360" id="cam-${i}"></canvas><div class="camera-placeholder" id="placeholder-${i}">SENSOR ${String(i+1).padStart(2,'0')}</div><span class="camera-index">C${String(i+1).padStart(2,'0')}</span><span class="camera-active"></span><span class="camera-name">${channelNames[i]}</span><span class="camera-detection-count" id="cam-count-${i}"></span>`;
  $('cameras').append(card);
}
for (let i=0;i<15;i++) {const bar=document.createElement('i'); $('capacity-bars').append(bar);}

async function control(data) {
  try {
    const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result=await response.json();
    if (!response.ok) throw new Error(result.error || t('控制请求失败'));
    state=result; updateState();
  } catch(error) {showError(error.message);}
}
function showError(message) {$('error').textContent=message; $('error').hidden=!message;}
function updateState() {
  if($('play')){
    $('play').textContent=state.playing?'Ⅱ':'▶';
    $('play').setAttribute('aria-label',t(state.playing?'暂停回放':'开始回放'));
  }
  if($('fps-select'))$('fps-select').value=String(state.fps || 8);
  $('traffic-select').value=String(state.load_mbps || 0);
  $('traffic-direction').value=state.load_direction || 'bidir';
  if($('time-current'))$('time-current').textContent=timeLabel(frameTime(state.index || 0));
  if($('time-total'))$('time-total').textContent=timeLabel(state.duration_s ?? ((state.frames || 39)-1)/2);
  $('scene-name').textContent=state.scene || 'scene-0061';
  $('scene-name').title=state.description || '';
  const scenes=state.scenes || [];
  const signature=currentLanguage()+scenes.map(s=>`${s.name}:${s.frames}`).join(',');
  if(signature!==sceneSignature) {
    sceneSignature=signature;
    $('scene-select').replaceChildren(...scenes.map(s=>new Option(t('{name} · {frames} 帧',{name:s.name,frames:s.frames}),s.name)));
  }
  $('scene-select').value=state.scene || 'scene-0061';
  $('dataset-caption').textContent=t('nuScenes · {frames} 配对帧 · {scenes} 场景',{frames:state.frames || 39,scenes:scenes.length || 1});
  $('viewer-caption').textContent=t(state.replay_location==='jetson'?'Jetson 本地浏览器 · 实时结果':'PC 浏览器 · 实时结果');
  $('fps-value').textContent=!state.playing ? '0.0' : state.actual_fps ? Number(state.actual_fps).toFixed(1):'—';
  $('latency-value').textContent=state.roundtrip_ms ? t('端到端 {ms} ms',{ms:Math.round(state.roundtrip_ms)}):t('等待首帧计算');
  $('input-size').textContent=formatBytes(state.input_bytes);
  if(document.activeElement!==$('gateway-url'))$('gateway-url').value=state.gateway_url || 'http://192.168.55.18:9000';
  if (state.error) showError(t('Jetson 处理端：')+state.error);
  else if (wsOpen) showError('');
  const connected=wsOpen && !!telemetry.connected;
  $('connection').classList.toggle('disconnected',!connected);
  $('connection').innerHTML=`<i></i>${t(connected?(state.playing?'实时回放':'回放暂停'):'等待 Jetson')}`;
  $('footer-status').textContent=connected?`JETSON CONNECTED · ${Number(state.processed || 0).toLocaleString()} FRAMES PROCESSED`:'WAITING FOR JETSON';
}
function updateTelemetry(t) {
  telemetry=t;
  updateNetwork(t);
  for (const [name,key] of [['gpu','gpu_percent'],['cpu','cpu_percent'],['memory','memory_percent']]) {
    const value=t[key];
    $(`${name}-value`).textContent=Number.isFinite(value)?`${name==='gpu'?value.toFixed(1):Math.round(value)}%`:'—';
    $(`${name}-bar`).style.width=`${Number.isFinite(value)?Math.min(100,value):0}%`;
  }
  $('memory-value').title=t.memory_used_gb?translateMemory(t):'';
  $('temperature').textContent=Number.isFinite(t.temperature_c)?`${t.temperature_c.toFixed(0)} °C`:'— °C';
  $('power').textContent=Number.isFinite(t.power_w)?`${t.power_w.toFixed(1)} W GPU`:'— W GPU';
  updateGpuNote();
  updateState();
}
function updateGpuNote() {
  if(latest?.vision_model) {
    $('model-detail').textContent=`${latest.vision_model} · 6 × ${latest.vision_resolution} · FP16`;
  }
  $('gpu-stages').hidden=!latest?.gpu_stages?.length;
  $('gpu-stages').title=latest?.gpu_stages?.join(' · ') || '';
  const window=telemetry.gpu_window_s;
  const valid=!telemetry.telemetry_stale && telemetry.connected && Number.isFinite(telemetry.gpu_percent);
  if(!valid) {
    $('gpu-value').textContent='—';$('gpu-bar').style.width='0%';
    $('gpu-note').textContent=t('GPU 监控暂不可用');
    $('gpu-value').title=telemetry.gpu_error || '';
    return;
  }
  const timing=latest?.timings.vision_cuda_ms;
  const roadTiming=latest?.timings.road_cuda_ms;
  $('gpu-note').textContent=latest?.surround?.semantic_lane_predictions && Number.isFinite(roadTiming)
    ?t('GPU {seconds} 秒均值 · 检测 {detection} ms · 路面分割 {road} ms',{seconds:window || 2,detection:timing.toFixed(1),road:roadTiming.toFixed(1)})
    :Number.isFinite(timing)
    ?t('GPU {seconds} 秒均值 · 六路推理 {ms} ms / 帧',{seconds:window || 2,ms:timing.toFixed(1)})
    :t('GPU {seconds} 秒均值 · 等待推理耗时',{seconds:window || 2});
  $('gpu-value').title=t('NVML 硬件采样 · 当前 {current}% · 窗口峰值 {peak}% · {interval} ms 读取间隔',{
    current:telemetry.gpu_instant_percent,peak:telemetry.gpu_peak_percent,interval:telemetry.gpu_sample_interval_ms});
}
function translateMemory(metrics) {return t('{used} / {total} GB 共享内存',{used:metrics.memory_used_gb,total:metrics.memory_total_gb});}
function updateNetwork(metrics) {
  const ports=metrics.ethernet_ports || [];
  const limit=Number(metrics.max_load_mbps || 900);
  if(loadLimit!==limit) {
    loadLimit=limit;
    const values=limit>1000?[0,1000,3000,6000,9000,9500]:[0,100,300,600,900];
    $('traffic-select').replaceChildren(...values.filter(v=>v<=limit).map(v=>new Option(v?`${v>=1000?v/1000+' Gbps':v+' Mbps'} / ${t('方向')}`:t('关闭 · 仅传感器流'),String(v))));
  } else {
    $('traffic-select').options[0].textContent=t('关闭 · 仅传感器流');
    for(const option of [...$('traffic-select').options].slice(1)) {const v=Number(option.value);option.textContent=`${v>=1000?v/1000+' Gbps':v+' Mbps'} / ${t('方向')}`;}
  }
  $('traffic-select').value=String(state.load_mbps || 0);
  const signature=ports.map(p=>p.interface).join(',');
  if(signature!==portSignature) {
    portSignature=signature;
    $('interface-select').replaceChildren(new Option(t('主数据链路'),'data'),...ports.map(p=>new Option(p.interface,p.interface)));
    if(monitor!=='data'&&!ports.some(p=>p.interface===monitor))monitor='data';
    $('interface-select').value=monitor;
  }
  $('interface-select').options[0].textContent=t('主数据链路');
  const selected=ports.find(p=>p.interface===monitor);
  const link=metrics.data_link || {input_mbps:metrics.tx_mbps,output_mbps:metrics.rx_mbps,input_bytes:metrics.tx_bytes,output_bytes:metrics.rx_bytes,capacity_mbps:10000,interface:metrics.interface || 'bev-host',kind:'software_ethernet',carrier:metrics.connected};
  const tx=Number((selected?selected.rx_mbps:link.input_mbps) || 0), rx=Number((selected?selected.tx_mbps:link.output_mbps) || 0);
  const capacity=selected?(selected.speed_mbps || 10000):(link.hardware_capacity_mbps || link.capacity_mbps || 10000);
  $('link-capacity').innerHTML=capacity>=1000?`${capacity/1000} <small>Gbps</small>`:`${capacity || '—'} <small>Mbps</small>`;
  $('link-description').textContent=`${selected?.interface || (link.peer_interface?link.interface+' ⇄ '+link.peer_interface:link.interface)} · ${selected?(selected.carrier?t('已连接'):t('未连接')):'FULL DUPLEX · TCP'}`;
  $('input-direction').textContent=t(selected?'接收 RX':'输入');
  $('output-direction').textContent=t(selected?'发送 TX':'输出');
  $('input-total-label').textContent=t(selected?'累计接收':'累计输入');
  $('output-total-label').textContent=t(selected?'累计发送':'累计输出');
  $('tx-rate').innerHTML=rateHTML(tx);
  $('rx-rate').innerHTML=rateHTML(rx);
  const utilization=capacity?Math.max(tx,rx)/capacity*100:0;
  $('load-value').textContent=capacity?utilization.toFixed(1):'—';
  [...$('capacity-bars').children].forEach((bar,i)=>bar.classList.toggle('on',i<Math.ceil(utilization/100*15)));
  $('tx-total').textContent=formatBytes(selected?selected.rx_bytes:link.input_bytes);
  $('rx-total').textContent=formatBytes(selected?selected.tx_bytes:link.output_bytes);
  $('link-led').classList.toggle('up',!!(selected?selected.carrier:link.carrier));
  $('ports-summary').textContent=t('{count} / {total} 已连接',{count:ports.filter(p=>p.carrier).length,total:ports.length});
  $('ethernet-ports').replaceChildren(...ports.map(port=>{
    const card=document.createElement('button');card.className='ethernet-port';card.classList.toggle('connected',port.carrier);card.classList.toggle('selected',monitor===port.interface);
    const title=document.createElement('span');title.className='port-title';title.textContent=port.interface;
    const status=document.createElement('span');status.className='port-status';status.textContent=port.carrier?(port.speed_mbps?`${port.speed_mbps/1000} GbE`:t('协商速率待检测')):t('未连接');
    const rate=document.createElement('span');rate.className='port-rates';rate.textContent=`RX ${rateLabel(Number(port.rx_mbps || 0))} · TX ${rateLabel(Number(port.tx_mbps || 0))}`;
    card.append(title,status,rate);card.title=`${port.mac} · ${port.operstate} · ${port.duplex} · MTU ${port.mtu || '—'} · ${port.namespace || 'host'}`;
    card.addEventListener('click',()=>selectMonitor(port.interface));return card;
  }));
  $('traffic-note').textContent=state.load_mbps ? t('传感器流＋{rate} TCP 压测 · {direction}',{rate:rateLabel(state.load_mbps),direction:t({bidir:'同时双向',input:'输入 TX',output:'输出 RX'}[state.load_direction || 'bidir'])}):t('仅传感器流 · 速率来自真实接口计数器');
  if(metrics.load_error)$('traffic-note').textContent=metrics.load_error;
  const local=state.replay_location==='jetson';
  const loopDescription=local
    ?'物理回环：{host} ⇄ {peer} · 10 GbE · MTU {mtu}。输入、结果和 TCP 压测都经过网线；数据回放与 UI 在 Jetson 本地运行。'
    :'物理回环：{host} ⇄ {peer} · 10 GbE · MTU {mtu}。输入、结果和双向 TCP 压测都经过网线；USB 用于 PC 管理和 UI。';
  const description=link.peer_interface?t(loopDescription,{host:link.interface,peer:link.peer_interface,mtu:link.mtu}):t(link.kind==='physical_ethernet'?'物理以太网 · {interface} · 读取该端口实际收发计数。':'J6015 板载 10 GbE · 当前数据走 {interface} 软件链路，双向限速 {capacity} Mbps；物理万兆口速率单独读取。',{interface:link.interface,capacity:link.transport_limit_mbps || link.capacity_mbps});
  $('network-source').textContent=description+' '+t(local?'本地回放接口：{interface}。压测作用于 GPU 数据链路。':'当前 PC 传输接口：{interface}。压测作用于内部 GPU 数据链路。',{interface:metrics.ingress_interface || (local?'lo':'usb0')});
  if(link.counter_source==='mac_hardware')$('network-source').textContent+=' '+t('MAC 硬件计数')+'.';
  if (metrics.sampled_at && rates.at(-1)?.time!==metrics.sampled_at) {
    rates.push({time:metrics.sampled_at,tx,rx}); if(rates.length>60)rates.shift();
  }
  drawNetworkChart();
}
function selectMonitor(interfaceName) {monitor=interfaceName;rates.length=0;$('interface-select').value=monitor;updateNetwork(telemetry);}

function drawCamera(i, data, version) {
  const image=new Image();
  image.onload=()=>{
    if (version!==frameVersion) return;
    const canvas=$(`cam-${i}`); canvas.width=data.width;canvas.height=data.height;
    const ctx=canvas.getContext('2d'); ctx.drawImage(image,0,0);
    if (projection) {
      for (const [x,y,depth] of data.projected_points) {
        const h=Math.round(165+Math.min(70,depth));
        ctx.fillStyle=`hsla(${h},92%,65%,0.8)`;
        ctx.fillRect(x-1.1,y-1.1,2.2,2.2);
      }
    }
    for (const detection of data.detections) {
      const [x1,y1,x2,y2]=detection.box, color=colors[detection.class_id] || '#62e9b5';
      ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      const label=`${className(detection.label)} ${(detection.score*100).toFixed(0)}%`;
      ctx.font='10px monospace';const width=ctx.measureText(label).width+7;
      ctx.fillStyle='#0c1526dc';ctx.fillRect(x1,Math.max(0,y1-14),width,14);
      ctx.fillStyle=color;ctx.fillText(label,x1+3,Math.max(10,y1-4));
    }
    $(`placeholder-${i}`).hidden=true;$(`cam-count-${i}`).textContent=`${data.detections.length} DET`;
  };
  image.src=data.image;
}
function showFrame(result,refreshOnly=false) {
  latest=result;if(!refreshOnly)lastFrame=Date.now();frameVersion++;
  const surroundButton=document.querySelector('[data-view="surround"]');
  const compareButton=document.querySelector('[data-view="compare"]');
  if(surroundButton)surroundButton.disabled=!result.images.surround;
  if(compareButton)compareButton.disabled=!result.images.surround;
  if(['surround','compare'].includes(view)&&!result.images.surround)changeView('fusion');
  $('viewer-loading').hidden=true;
  $('points-value').textContent=result.points_count.toLocaleString();
  $('detections-value').textContent=result.detections.length;
  $('vision-count').textContent=t('{count} 个视觉检测 → 空间关联',{count:result.vision_detections});
  $('object-count').textContent=result.detections.length;
  $('sensor-frame').textContent=`SYNC · FRAME ${String(result.index+1).padStart(3,'0')}`;
  $('lidar-caption').textContent=t('{count} 点 · 标定对齐六路视觉',{count:result.points_count.toLocaleString()});
  $('pipeline-duration').textContent=t('流水线 {ms} ms',{ms:Math.round(result.timings.total_ms)});
  $('pipeline-duration').title=Object.entries(result.timings).map(([k,v])=>`${k}: ${v} ms`).join('\n');
  $('gpu-pipeline').textContent=`VISION ${Math.round(result.timings.vision_ms)} ms`;
  updateGpuNote();
  if($('time-current'))$('time-current').textContent=timeLabel(frameTime(result.index));
  if(section==='sensors')result.cameras.forEach((data,i)=>drawCamera(i,data,frameVersion));
  if(section==='fusion') {
    for (const key of ['lidar','vision','fusion']) $(`inter-${key}`).src=result.images[key];
    renderFusionFrame(result,frameVersion);
  }
  $('object-dots').replaceChildren();
  result.detections.slice(0,20).forEach(d=>{const dot=document.createElement('i');dot.style.background=colors[d.class_id];$('object-dots').append(dot);});
  if(section==='fusion') {
    $('object-list').replaceChildren();
    if(!result.detections.length)$('object-list').innerHTML=`<div class="empty-state">${t('本帧暂无满足雷达关联条件的目标')}</div>`;
    result.detections.slice().sort((a,b)=>b.score-a.score).forEach(d=>{
      const item=document.createElement('div');item.className='object-row';
      const distance=Math.hypot(d.center[0],d.center[1]);
      item.innerHTML=`<span class="object-marker" style="color:${colors[d.class_id]};border-color:${colors[d.class_id]}50">${String(d.id).padStart(2,'0')}</span><div><div class="object-name">${className(d.label)}<small>#${d.id}</small></div><div class="object-info">${distance.toFixed(1)} m · ${d.points} LiDAR points</div></div><span class="object-confidence">${(d.score*100).toFixed(0)}%</span>`;
      $('object-list').append(item);
    });
  }
  if(!refreshOnly){fpsSamples.push(Number(state.actual_fps || 0));if(fpsSamples.length>30)fpsSamples.shift();}
  drawTinyChart();updateState();
}
function renderFusionFrame(result,version) {
  const requestedView=view;
  const commit=()=>{
    if(version!==frameVersion||section!=='fusion'||view!==requestedView)return;
    displayedFusion=result;
    if(view!=='cloud')$('bev-view').src=result.images[bevKey()];
    updatePointCloud(result);
    const frame=`FRAME ${String(result.index+1).padStart(3,'0')} / ${String(state.frames || 39).padStart(3,'0')}`;
    for(const id of ['view-frame','cloud-frame'])$(id).textContent=frame;
    $('bev-view').dataset.frame=String(result.index);
    $('three-view').dataset.frame=String(result.index);
    updateViewExplanation(result);
  };
  if(view==='compare') {
    // Decode the BEV before committing both panes, so the point cloud cannot
    // advance while the paired image is still loading.
    const image=new Image();image.onload=commit;image.src=result.images.surround;
  } else commit();
}

function canvasDimensions(canvas) {
  const rect=canvas.getBoundingClientRect(), dpr=Math.min(2,devicePixelRatio);
  const w=Math.max(1,Math.round(rect.width)),h=Math.max(1,Math.round(rect.height));
  canvas.width=w*dpr;canvas.height=h*dpr;
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);return {ctx,w,h};
}
function drawNetworkChart() {
  if(section!=='network' || !$('network-chart').getBoundingClientRect().width)return;
  const {ctx,w,h}=canvasDimensions($('network-chart'));
  const max=Math.max(20,...rates.flatMap(p=>[p.tx,p.rx]))*1.08;
  $('chart-max').textContent=max>=1000?`${(max/1000).toFixed(1)} Gbps`:`${Math.ceil(max/10)*10} Mbps`;
  ctx.strokeStyle='#233a44';ctx.lineWidth=.5;
  for(let i=1;i<=3;i++){ctx.beginPath();ctx.moveTo(0,h*i/4);ctx.lineTo(w,h*i/4);ctx.stroke();}
  for(const [key,color] of [['tx','#62e9b5'],['rx','#69b7ff']]) {
    const coords=rates.map((p,i)=>[w*(60-rates.length+i)/59,h-5-(h-10)*p[key]/max]);
    if(!coords.length)continue;
    ctx.beginPath();coords.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));
    ctx.lineWidth=1.4;ctx.strokeStyle=color;ctx.stroke();
    ctx.lineTo(coords.at(-1)[0],h);ctx.lineTo(coords[0][0],h);ctx.closePath();
    const gradient=ctx.createLinearGradient(0,0,0,h);gradient.addColorStop(0,color+'27');gradient.addColorStop(1,color+'02');ctx.fillStyle=gradient;ctx.fill();
  }
}
function drawTinyChart() {
  const {ctx,w,h}=canvasDimensions($('fps-chart'));const max=Math.max(5,...fpsSamples)*1.1;
  ctx.beginPath();fpsSamples.forEach((v,i)=>{const x=w*i/Math.max(1,fpsSamples.length-1),y=h-2-v/max*(h-5);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.lineWidth=1.3;ctx.strokeStyle='#62e9b5';ctx.stroke();
}

// A WebGL scene displays the actual returned ego-coordinate point cloud and estimated boxes.
const scene=new THREE.Scene();scene.fog=new THREE.FogExp2('#0b151c',.006);
const camera=new THREE.PerspectiveCamera(43,1,.1,300);
let renderer;
try {renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});}
catch(error){showError(t('浏览器 WebGL 不可用，请使用融合 BEV 视图。'));}
if(renderer){renderer.setPixelRatio(Math.min(2,devicePixelRatio));$('three-view').append(renderer.domElement);}
const grid=new THREE.GridHelper(100,20,'#31564f','#18303a');scene.add(grid);
const axesGeometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,.02,-50),new THREE.Vector3(0,.02,50),new THREE.Vector3(-50,.02,0),new THREE.Vector3(50,.02,0)]);
scene.add(new THREE.LineSegments(axesGeometry,new THREE.LineBasicMaterial({color:'#325459',transparent:true,opacity:.65})));
const maxPoints=60000, positions=new Float32Array(maxPoints*3), pointColors=new Float32Array(maxPoints*3);
const cloudGeometry=new THREE.BufferGeometry();cloudGeometry.setAttribute('position',new THREE.BufferAttribute(positions,3));cloudGeometry.setAttribute('color',new THREE.BufferAttribute(pointColors,3));cloudGeometry.setDrawRange(0,0);
const cloud=new THREE.Points(cloudGeometry,new THREE.PointsMaterial({size:.19,vertexColors:true,transparent:true,opacity:.88,sizeAttenuation:true}));scene.add(cloud);
const egoGeometry=new THREE.BoxGeometry(1.9,1.45,4.4);
const ego=new THREE.LineSegments(new THREE.EdgesGeometry(egoGeometry),new THREE.LineBasicMaterial({color:'#d4e8e4',transparent:true,opacity:.85}));ego.position.set(0,.75,0);scene.add(ego);
const egoFill=new THREE.Mesh(egoGeometry,new THREE.MeshBasicMaterial({color:'#3a6d62',transparent:true,opacity:.15}));egoFill.position.copy(ego.position);scene.add(egoFill);
const egoForward=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(0,.06,3),3,'#62e9b5',.7,.4);scene.add(egoForward);
const boxes=new THREE.Group();scene.add(boxes);let labels=[];
let azimuth=2.6, polar=.91, radius=67, mapAzimuth=.4, mapPolar=.72, mapRadius=95, dragging=null,lastPointer=null,autoRotate=true;
const bevKey=()=>view==='compare'?'surround':view;
const cloudVisible=()=>view==='compare'||view==='map';
const bevTransforms=new Map();
const bevTransform=()=>{
  const key=bevKey();
  if(!bevTransforms.has(key))bevTransforms.set(key,{zoom:1,x:0,y:0});
  return bevTransforms.get(key);
};
function applyBevTransform() {
  if(view==='cloud')return;
  const position=bevTransform(),rect=$('viewer').getBoundingClientRect();
  if(rect.width<=0 || rect.height<=0)return;
  // object-fit:contain centers a square BEV within the viewer.
  const side=Math.min(rect.width,rect.height)*position.zoom;
  const limitX=Math.max(0,(side-rect.width)/2),limitY=Math.max(0,(side-rect.height)/2);
  position.x=Math.max(-limitX,Math.min(limitX,position.x));
  position.y=Math.max(-limitY,Math.min(limitY,position.y));
  $('bev-view').style.transform=`translate(${position.x}px,${position.y}px) scale(${position.zoom})`;
  $('viewer-scale').textContent=`${bevKey()==='surround'?'60 m × 60 m':'100 m × 100 m'} · ${position.zoom.toFixed(1)}×`;
}
function resetCloud() {
  dragging=null;
  if(view==='map'){mapAzimuth=.4;mapPolar=.72;mapRadius=160;}
  else {azimuth=2.6;polar=.91;radius=67;autoRotate=true;}
}
function resetView() {
  dragging=null;
  if(view==='cloud')resetCloud();
  else {bevTransforms.set(bevKey(),{zoom:1,x:0,y:0});applyBevTransform();}
}
function updatePointCloud(result) {
  const source=view==='map'?result.map_pointcloud:result.pointcloud;
  const count=Math.min(maxPoints,source?.length || 0);
  for(let i=0;i<count;i++) {
    const [x,y,z,color]=source[i];positions[i*3]=-y;positions[i*3+1]=z;positions[i*3+2]=x;
    pointColors[i*3]=((color>>16)&255)/255;pointColors[i*3+1]=((color>>8)&255)/255;pointColors[i*3+2]=(color&255)/255;
  }
  cloudGeometry.setDrawRange(0,count);cloudGeometry.attributes.position.needsUpdate=true;cloudGeometry.attributes.color.needsUpdate=true;cloudGeometry.computeBoundingSphere();
  while(boxes.children.length){const child=boxes.children[0];boxes.remove(child);child.geometry.dispose();child.material.dispose();}
  labels=[];$('labels-3d').replaceChildren();
  const pose=result.mapping?.ego_position_m || [0,0,0];
  const anchor=view==='map'?new THREE.Vector3(-pose[1],0,pose[0]):new THREE.Vector3(0,0,0);
  const detections=view==='map'?[]:result.detections;
  ego.visible=egoFill.visible=egoForward.visible=view!=='map';
  grid.position.copy(anchor);boxes.position.copy(anchor);
  detections.forEach(d=>{
    const [x,y,z]=d.center,[sx,sy,sz]=d.size;
    const geometry=new THREE.BoxGeometry(sy,sz,sx),edges=new THREE.EdgesGeometry(geometry);geometry.dispose();
    const box=new THREE.LineSegments(edges,new THREE.LineBasicMaterial({color:colors[d.class_id],transparent:true,opacity:.88}));
    box.position.set(-y,z,x);boxes.add(box);
    if(labels.length<18){const element=document.createElement('span');element.className='object-label';element.style.color=colors[d.class_id];element.textContent=`${className(d.label).toUpperCase()} #${d.id}`;$('labels-3d').append(element);labels.push({element,position:new THREE.Vector3(-y,z+sz/2+.5,x)});}
  });
}
function resize3D() {
  const {width,height}=$('cloud-pane').getBoundingClientRect();
  if(renderer && width>0 && height>0){renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();}
  drawNetworkChart();drawTinyChart();
  applyBevTransform();
}
new ResizeObserver(resize3D).observe($('viewer'));
new ResizeObserver(resize3D).observe($('cloud-pane'));
new ResizeObserver(drawNetworkChart).observe($('network-chart'));
function animate() {
  requestAnimationFrame(animate);
  if(!renderer || section!=='fusion' || !cloudVisible() || document.hidden) return;
  const mapView=view==='map';
  if(autoRotate&&!mapView&&dragging!=='cloud'&&!matchMedia('(prefers-reduced-motion: reduce)').matches)azimuth+=.00018;
  const yaw=mapView?mapAzimuth:azimuth,pitch=mapView?mapPolar:polar,distance=mapView?mapRadius:radius;
  const pose=latest?.mapping?.ego_position_m || [0,0,0];
  const target=mapView?new THREE.Vector3(-pose[1],0,pose[0]):new THREE.Vector3(0,0,5);
  camera.position.set(target.x+distance*Math.sin(pitch)*Math.sin(yaw),target.y+distance*Math.cos(pitch),target.z+distance*Math.sin(pitch)*Math.cos(yaw));
  camera.lookAt(target);
  renderer.render(scene,camera);
  // Mirror label positions with the canvas while keeping their text readable.
  for(const label of labels){const p=label.position.clone().project(camera);const visible=p.z<1&&Math.abs(p.x)<.92&&Math.abs(p.y)<.85;label.element.hidden=!visible;if(visible){label.element.style.left=`${(1-p.x)/2*100}%`;label.element.style.top=`${(1-p.y)/2*100}%`;}}
}
animate();
for(const [id,kind] of [['viewer','bev'],['cloud-pane','cloud']]) {
  const pane=$(id);
  pane.addEventListener('pointerdown',e=>{
    if(e.button!==0||e.target.closest('button'))return;
    e.preventDefault();dragging=kind;if(kind==='cloud')autoRotate=false;
    lastPointer={x:e.clientX,y:e.clientY};pane.setPointerCapture(e.pointerId);
  });
  pane.addEventListener('pointermove',e=>{
    if(dragging!==kind)return;
    const dx=e.clientX-lastPointer.x,dy=e.clientY-lastPointer.y;
    if(kind==='cloud'){
      if(view==='map'){mapAzimuth+=dx*.007;mapPolar=Math.max(.08,Math.min(1.52,mapPolar+dy*.005));}
      else {azimuth+=dx*.007;polar=Math.max(.12,Math.min(1.45,polar+dy*.005));}
    }
    else {const position=bevTransform();position.x+=dx;position.y+=dy;applyBevTransform();}
    lastPointer={x:e.clientX,y:e.clientY};
  });
  for(const event of ['pointerup','pointercancel','lostpointercapture'])pane.addEventListener(event,()=>{if(dragging===kind)dragging=null;});
}
$('viewer').addEventListener('wheel',e=>{
  e.preventDefault();
  const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?$('viewer').clientHeight:1);
  const position=bevTransform(),rect=$('viewer').getBoundingClientRect();
  const zoom=Math.max(1,Math.min(8,position.zoom*Math.exp(-delta*.002)));
  const ratio=zoom/position.zoom,anchorX=e.clientX-rect.left-rect.width/2,anchorY=e.clientY-rect.top-rect.height/2;
  position.x=anchorX-(anchorX-position.x)*ratio;
  position.y=anchorY-(anchorY-position.y)*ratio;
  position.zoom=zoom;applyBevTransform();
},{passive:false});
$('viewer').addEventListener('dblclick',resetView);
$('cloud-pane').addEventListener('wheel',e=>{
  e.preventDefault();
  const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?$('cloud-pane').clientHeight:1);
  if(view==='map')mapRadius=Math.max(25,Math.min(800,mapRadius+delta*.05));
  else radius=Math.max(20,Math.min(130,radius+delta*.05));
},{passive:false});
$('cloud-pane').addEventListener('dblclick',e=>{if(!e.target.closest('button'))resetCloud();});
function updateViewExplanation(result=displayedFusion || latest) {
  const key=bevKey();
  $('view-label').textContent=t({surround:'车道线鸟瞰',fusion:'几何融合 BEV',map:'三维位姿累积',lidar:'雷达 BEV',vision:'视觉空间落点'}[key] || '3D 点云');
  const semantic='蓝灰：道路 · 浅黄：车道线 · 彩色：物体轮廓 · 滚轮缩放、拖动平移、双击复位';
  const notes={
    compare:'两侧来自同一后端帧 · 左侧滚轮缩放、拖动平移 · 右侧滚轮缩放、拖动旋转 · 双击分别复位',
    surround:semantic,
    cloud:'真实雷达点云与几何估计框 · 滚轮缩放、拖动旋转、双击复位',
    fusion:'视觉检测与标定雷达关联，估计空间框并叠加到雷达占用图；当前为几何融合，未运行 BEVFusion 特征融合网络。',
    map:'后端按位姿累积当前场景的全部三维雷达，可拖动旋转、滚轮缩放、双击复位；未运行自主定位 SLAM，动态物体可能留痕。',
    lidar:'本帧雷达点的高度与密度栅格；只保留距地面 -0.6 至 6 米的点。',
    vision:'六路视觉检测经雷达测距后的空间框；没有足够雷达点支持的检测不生成空间框。'
  };
  $('view-note').textContent=t(notes[view],{frames:result?.map_frames || 0,limit:result?.mapping?.max_frames || 25});
  const f=result?.fusion_bev,m=result?.mapping;
  if(!f||!m){$('compute-evidence').textContent=t('等待后端计算结果');return;}
  $('compute-evidence').textContent=t('Jetson 帧 {frame} · 占用 {cells} 格 · 关联 {points} 点 / {objects} 目标 · 累积 {frames} 帧 / {mapPoints} 点 · 融合 {fusionMs} ms · 累积 {mapMs} ms',{
    frame:result.index+1,cells:f.occupied_cells,points:f.associated_lidar_points,objects:f.object_count,
    frames:m.frames,mapPoints:m.retained_points,fusionMs:result.timings.fusion_ms,mapMs:result.timings.mapping_ms
  });
}
function changeView(key) {
  if(!['compare','fusion','map'].includes(key))return;
  dragging=null;
  view=key;$('viewports').classList.toggle('compare',key==='compare');
  $('viewer').hidden=key==='map';$('cloud-pane').hidden=!cloudVisible();
  $('bev-view').alt=t(bevKey()==='surround'?'道路、车道线与物体掩码鸟瞰':'GPU 计算的鸟瞰占用与检测图');
  if(latest&&section==='fusion')renderFusionFrame(latest,frameVersion);
  updateViewExplanation();
  applyBevTransform();
  requestAnimationFrame(resize3D);
  document.querySelectorAll('.view-tabs button').forEach(button=>{const selected=button.dataset.view===key;button.classList.toggle('selected',selected);button.setAttribute('aria-selected',selected);});
  document.querySelectorAll('.intermediates button').forEach(button=>{const selected=button.dataset.view===key;button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));});
}
function changeSection(key) {
  if(!['sensors','fusion','network'].includes(key))return;
  section=key;
  document.querySelectorAll('[data-section]').forEach(button=>{
    const selected=button.dataset.section===key;
    button.classList.toggle('selected',selected);
    button.setAttribute('aria-selected',String(selected));
    button.tabIndex=selected?0:-1;
    $(button.getAttribute('aria-controls')).hidden=!selected;
  });
  document.querySelectorAll('[data-nav]').forEach(button=>{
    const selected=button.dataset.nav===key;
    button.classList.toggle('active',selected);
    button.setAttribute('aria-pressed',String(selected));
  });
  try {localStorage.setItem('thor-section',key);} catch {}
  if(latest)showFrame(latest,true);
  requestAnimationFrame(resize3D);
}
const sectionTabs=[...document.querySelectorAll('[data-section]')];
sectionTabs.forEach((button,index)=>{
  button.addEventListener('click',()=>changeSection(button.dataset.section));
  button.addEventListener('keydown',event=>{
    let next;
    if(event.key==='ArrowRight')next=(index+1)%sectionTabs.length;
    if(event.key==='ArrowLeft')next=(index+sectionTabs.length-1)%sectionTabs.length;
    if(event.key==='Home')next=0;
    if(event.key==='End')next=sectionTabs.length-1;
    if(next===undefined)return;
    event.preventDefault();
    sectionTabs[next].focus();changeSection(sectionTabs[next].dataset.section);
  });
});
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>changeView(button.dataset.view)));
$('reset-camera').addEventListener('click',resetView);
$('reset-cloud').addEventListener('click',resetCloud);
if($('play'))$('play').addEventListener('click',()=>control({playing:!state.playing}));
if($('step'))$('step').addEventListener('click',()=>control({step:true}));
if($('fps-select'))$('fps-select').addEventListener('change',e=>control({fps:Number(e.target.value)}));
$('traffic-select').addEventListener('change',e=>control({load_mbps:Number(e.target.value)}));
$('traffic-direction').addEventListener('change',e=>control({load_direction:e.target.value}));
$('interface-select').addEventListener('change',e=>selectMonitor(e.target.value));
$('gateway-apply').addEventListener('click',()=>control({gateway:$('gateway-url').value}));
$('scene-select').addEventListener('change',e=>control({scene:e.target.value}));
$('language-select').addEventListener('change',e=>{setLanguage(e.target.value);updateTelemetry(telemetry);updateViewExplanation();if(latest)showFrame(latest,true);});
$('projection-toggle').addEventListener('click',()=>{projection=!projection;$('projection-toggle').classList.toggle('active',projection);$('projection-toggle').setAttribute('aria-pressed',projection);if(latest)latest.cameras.forEach((data,i)=>drawCamera(i,data,frameVersion));});
function setFullscreenLayout(active) {
  document.documentElement.classList.toggle('is-fullscreen',active);
  $('fullscreen').title=t(active?'退出全屏':'全屏展示');
  $('fullscreen').setAttribute('aria-label',t(active?'退出全屏':'全屏展示'));
  window.scrollTo(0,0);
  requestAnimationFrame(resize3D);
}
$('fullscreen').addEventListener('click',async()=>{
  const active=document.documentElement.classList.contains('is-fullscreen');
  setFullscreenLayout(!active);
  try {
    if(document.fullscreenElement)await document.exitFullscreen();
    else if(!active)await document.documentElement.requestFullscreen();
  } catch { /* Keep the expanded layout when browser fullscreen is unavailable. */ }
});
document.addEventListener('fullscreenchange',()=>{
  setFullscreenLayout(!!document.fullscreenElement);
});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!document.fullscreenElement)setFullscreenLayout(false);});
document.querySelectorAll('[data-nav]').forEach(button=>button.addEventListener('click',()=>{
  changeSection(button.dataset.nav);
  document.querySelector('.data-tabs').scrollIntoView({behavior:'smooth',block:'start'});
}));
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();control({playing:!state.playing});}if(e.code==='ArrowRight')control({step:true});});

function connect() {
  const socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
  socket.onopen=()=>{wsOpen=true;updateState();};
  socket.onmessage=event=>{try{const data=JSON.parse(event.data);if(data.state)state=data.state;if(data.telemetry)updateTelemetry(data.telemetry);if(data.type==='frame')showFrame(data.result);else updateState();}catch(error){showError(t('界面数据解析失败：')+error.message);}};
  socket.onclose=()=>{wsOpen=false;updateState();setTimeout(connect,1500);};
  socket.onerror=()=>socket.close();
}
changeView(view);
changeSection(section);
connect();
setInterval(()=>{if(lastFrame&&state.playing&&Date.now()-lastFrame>10000&&!state.error)showError(t('处理结果暂时中断，正在等待新的 Jetson 帧。'));},3000);
