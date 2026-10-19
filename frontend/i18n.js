const translations = {
  '退出全屏':'Exit fullscreen',
  '鸟瞰＋3D 对比':'BEV + 3D comparison',
  '三维位姿累积':'3D pose-aligned accumulation',
  '几何融合 BEV':'Geometric fusion BEV',
  '位姿累积图':'Pose-aligned accumulation',
  '等待后端计算结果':'Waiting for backend results',
  '两侧来自同一后端帧 · 左侧滚轮缩放、拖动平移 · 右侧滚轮缩放、拖动旋转 · 双击分别复位':'Both panes use the same backend frame · Left: scroll to zoom, drag to pan · Right: scroll to zoom, drag to rotate · Double-click either pane to reset',
  '真实雷达点云与几何估计框 · 滚轮缩放、拖动旋转、双击复位':'Real LiDAR points and estimated boxes · Scroll to zoom, drag to rotate, double-click to reset',
  '视觉检测与标定雷达关联，估计空间框并叠加到雷达占用图；当前为几何融合，未运行 BEVFusion 特征融合网络。':'Camera detections are associated with calibrated LiDAR; estimated boxes overlay LiDAR occupancy. This is geometric fusion; no BEVFusion feature fusion network is running.',
  '按数据集提供的位姿对齐最近 {frames} / {limit} 帧雷达，在场景首帧坐标系累积；未运行自主定位 SLAM，动态物体可能留痕。':'Latest {frames} / {limit} LiDAR frames aligned using dataset poses in the first ego frame. No autonomous SLAM is running; moving objects may leave trails.',
  '后端已按位姿对齐最近 {frames} / {limit} 帧三维雷达，可拖动旋转、滚轮缩放、双击复位；未运行自主定位 SLAM，动态物体可能留痕。':'The backend pose-aligns the latest {frames} / {limit} 3D LiDAR frames. Drag to rotate, scroll to zoom, and double-click to reset. No autonomous SLAM is running; moving objects may leave trails.',
  '本帧雷达点的高度与密度栅格；只保留距地面 -0.6 至 6 米的点。':'Height and density grid from this LiDAR frame; keeps points at -0.6 to 6 meters in ego coordinates.',
  '六路视觉检测经雷达测距后的空间框；没有足够雷达点支持的检测不生成空间框。':'Six-camera detections ranged by LiDAR; detections without enough LiDAR support produce no spatial box.',
  'Jetson 帧 {frame} · 占用 {cells} 格 · 关联 {points} 点 / {objects} 目标 · 累积 {frames} 帧 / {mapPoints} 点 · 融合 {fusionMs} ms · 累积 {mapMs} ms':'Jetson frame {frame} · {cells} occupied cells · {points} associated points / {objects} objects · {frames} accumulated frames / {mapPoints} points · Fusion {fusionMs} ms · Accumulation {mapMs} ms',
  '车道线鸟瞰':'Semantic BEV',
  '重置当前视角':'Reset current view',
  '蓝灰：道路 · 浅黄：车道线 · 彩色：物体轮廓 · 滚轮缩放、拖动平移、双击复位':'Blue-gray: road · Pale yellow: lanes · Colored: objects · Scroll to zoom, drag to pan, double-click to reset',
  '道路、车道线与物体掩码鸟瞰':'Road, lane and object mask BEV',
  '蓝灰：道路 · 浅黄：车道线 · 彩色：物体轮廓 · 暗色：盲区或不可靠区域':'Blue-gray: road · Pale yellow: lanes · Colored: object footprints · Dark: blind or unreliable areas',
  '标定相机路面鸟瞰与空间检测':'Calibrated camera road BEV and spatial detections',
  '车道线来自真实相机影像 · 平面投影会拉伸车辆等立体物体':'Road markings from real camera images · Ground projection stretches raised objects',
  'Jetson 本地浏览器 · 实时结果':'Jetson local browser · Live results',
  '物理回环：{host} ⇄ {peer} · 10 GbE · MTU {mtu}。输入、结果和 TCP 压测都经过网线；数据回放与 UI 在 Jetson 本地运行。':'Physical loop: {host} ⇄ {peer} · 10 GbE · MTU {mtu}. Inputs, results and TCP load use the cable; replay and UI run locally on the Jetson.',
  '本地回放接口：{interface}。压测作用于 GPU 数据链路。':'Local replay interface: {interface}. Extra load runs on the GPU data link.',
  'GPU 加速：图像 · 检测 · 点云 · 建图':'GPU accelerated: images · detection · LiDAR · mapping',
  'GPU 监控暂不可用':'GPU monitor unavailable',
  'GPU {seconds} 秒均值 · 检测 {detection} ms · 路面分割 {road} ms':'GPU {seconds}s average · Detection {detection} ms · Road segmentation {road} ms',
  'GPU {seconds} 秒均值 · 六路推理 {ms} ms / 帧':'GPU {seconds}s average · Six-camera inference {ms} ms / frame',
  'GPU {seconds} 秒均值 · 等待推理耗时':'GPU {seconds}s average · Waiting for inference timing',
  'NVML 硬件采样 · 当前 {current}% · 窗口峰值 {peak}% · {interval} ms 读取间隔':'NVML hardware samples · Latest {current}% · Window peak {peak}% · Polled every {interval} ms',
  '融合与建图':'Fusion & mapping','网络与算力':'Network & compute',
  '六路相机与雷达投影':'Six cameras with LiDAR overlays',
  '点云、BEV 与空间检测':'Point cloud, BEV and detections',
  '网口速率与 GPU 负载':'Ethernet throughput and GPU load',
  '数据视图':'Data views','回放控制':'Replay controls','同步回放':'Synchronized replay',
  '链路测试设置':'Link test settings',
  'Thor Fusion 首页':'Thor Fusion home',
  '融合总览':'Fusion overview','传感器输入':'Sensor inputs','链路监控':'Network monitor',
  '连接中':'Connecting','全屏展示':'Fullscreen','让每一束数据，':'Every sensor stream. ',
  '看见空间。':'One spatial view.','六路视觉 × 激光雷达':'Six cameras × LiDAR',
  'Jetson Thor GPU 融合':'Jetson Thor GPU fusion','实时 BEV 建图':'Live BEV mapping',
  '连续配对传感器回放':'Continuous paired sensor replay','选择场景':'Select scene',
  '感知处理速率':'Perception throughput','等待首帧计算':'Waiting for first frame',
  '当前点云':'LiDAR point cloud','真实 LiDAR 扫描':'Real LiDAR scans',
  '融合空间目标':'Fused spatial objects','六路视觉＋雷达测距':'Six cameras + LiDAR ranging',
  '以太网链路利用率':'Ethernet link utilization','双向链路实测':'Measured bidirectional traffic',
  '等待配对帧':'Waiting for paired frame','雷达投影':'LiDAR projection',
  '加载真实激光雷达扫描':'Loading real LiDAR scans','输入数据包':'Input packet','GPU 处理端':'GPU worker',
  '空间融合视图':'Spatial fusion','3D 点云':'3D point cloud','融合 BEV':'Fused BEV','累积建图':'Accumulated map',
  '重置视角':'Reset view','重置三维视角':'Reset 3D view',
  'GPU 计算的鸟瞰占用与检测图':'GPU bird’s-eye occupancy and detections',
  '车辆':'Car','行人':'Person','空间点云':'Point cloud','正在建立空间感知':'Building spatial perception',
  '真实六路影像与点云，正在 Jetson GPU 上处理':'Processing six real camera views and LiDAR on Jetson GPU',
  '暂停回放':'Pause replay','开始回放':'Start replay','单帧前进':'Next frame',
  '场景回放进度':'Replay position','回放':'Replay','回放速率':'Replay rate',
  '处理中间结果':'Intermediate results','点云投影形成的 BEV 高度与密度图':'LiDAR BEV height and density',
  '视觉检测与标定点云关联后的空间落点':'Camera detections associated with calibrated LiDAR',
  '融合检测结果鸟瞰图':'Fused detections in bird’s-eye view','雷达 BEV':'LiDAR BEV',
  '视觉空间落点':'Vision spatial anchors','融合检测':'Fused detections',
  '以太网传输速率':'Ethernet throughput','监控接口':'Monitor interface','选择网口':'Select network interface',
  '主数据链路':'Primary data link','输入':'Input','输出':'Output','60 秒实时窗口':'60-second live window',
  '累计输入':'Total input','累计输出':'Total output','Jetson 有线网口':'Jetson Ethernet ports',
  '附加双向链路压测':'Extra bidirectional load','附加双向 TCP 压测速率':'Extra TCP load per direction',
  '关闭 · 仅传感器流':'Off · sensor stream only','100 Mbps / 方向':'100 Mbps / direction',
  '300 Mbps / 方向':'300 Mbps / direction','600 Mbps / 方向':'600 Mbps / direction',
  '900 Mbps / 方向':'900 Mbps / direction','速率来自真实接口计数器':'Rates from live interface counters',
  '链路详情':'Link details','THOR 算力监控':'THOR compute monitor','内存':'RAM',
  '空间检测目标':'Spatial detections','等待视觉与点云关联结果':'Waiting for camera–LiDAR associations',
  '颜色表示类别 · 3D 框为雷达关联几何估计':'Class colors · 3D boxes estimated from LiDAR associations',
  '数据流全链路':'End-to-end data flow','真实数据 · 实际 GPU 计算 · 实际流量计数':'Real data · GPU inference · Measured traffic',
  '六目＋LiDAR':'Six cameras + LiDAR','nuScenes · 39 配对帧':'nuScenes · 39 paired frames',
  '以太网输入':'Ethernet input','TCP 传感器数据流':'TCP sensor stream','以太网输出':'Ethernet output',
  '图像 · 点云 · BEV · 检测':'Images · Points · BEV · Objects','可视化界面':'Visualization UI',
  'PC 浏览器 · 实时结果':'PC browser · Live results',
  'FP16 视觉检测 · 标定点云融合 · GPU 占用建图':'FP16 vision · Calibrated fusion · GPU occupancy mapping',
  '连接 Jetson 处理中':'Connecting to Jetson',
  '控制请求失败':'Control request failed','端到端 {ms} ms':'End-to-end {ms} ms',
  'nuScenes · {frames} 配对帧 · {scenes} 场景':'nuScenes · {frames} paired frames · {scenes} scenes',
  'Jetson 处理端：':'Jetson worker: ','实时回放':'Live replay','回放暂停':'Replay paused','等待 Jetson':'Waiting for Jetson',
  '{used} / {total} GB 共享内存':'{used} / {total} GB shared memory',
  '传感器流＋{rate} Mbps / 方向 TCP 压测':'Sensor stream + {rate} Mbps / direction TCP load',
  '仅传感器流 · 速率来自真实接口计数器':'Sensor stream only · Live interface counters',
  '{count} 个视觉检测 → 空间关联':'{count} vision detections → spatial association',
  '{count} 点 · 标定对齐六路视觉':'{count} points · Calibrated to six cameras',
  '流水线 {ms} ms':'Pipeline {ms} ms',
  '本帧暂无满足雷达关联条件的目标':'No LiDAR-associated objects in this frame',
  '浏览器 WebGL 不可用，请使用融合 BEV 视图。':'WebGL unavailable. Please use the fused BEV view.',
  '界面数据解析失败：':'UI data parsing failed: ',
  '处理结果暂时中断，正在等待新的 Jetson 帧。':'Waiting for new processing results from Jetson.',
  '未连接':'No link','已连接':'Connected','{count} / {total} 已连接':'{count} / {total} connected',
  '协商速率待检测':'Link speed unavailable','端口未启用':'Port inactive',
  '累计接收':'Total RX','累计发送':'Total TX','接收 RX':'Receive RX','发送 TX':'Transmit TX',
  '软件 TCP 链路 · {interface} · 双向 {capacity} Mbps 限速；物理网口速率单独读取。':'Software TCP link · {interface} · {capacity} Mbps per direction; physical ports are measured separately.',
  '物理以太网 · {interface} · 读取该端口实际收发计数。':'Physical Ethernet · {interface} · Live RX/TX counters for this port.',
  '当前 PC 传输接口：{interface}。压测作用于内部 GPU 数据链路。':'Current PC transport interface: {interface}. Extra load runs on the internal GPU data link.',
  '全部场景':'All scenes',
  'Jetson 传输地址':'Jetson transport address','应用':'Apply',
  '{name} · {frames} 帧':'{name} · {frames} frames',
  '方向':'direction',
  '压测方向':'Load direction','同时双向':'Bidirectional','输入 TX':'Input TX','输出 RX':'Output RX',
  '附加网口压测':'Extra Ethernet load','附加 TCP 压测速率':'Extra TCP load rate',
  'MAC 硬件计数':'MAC hardware counters',
  '传感器流＋{rate} TCP 压测 · {direction}':'Sensor stream + {rate} TCP load · {direction}',
  '传感器流＋{rate} / 方向 TCP 压测':'Sensor stream + {rate} / direction TCP load',
  '物理回环：{host} ⇄ {peer} · 10 GbE · MTU {mtu}。输入、结果和双向 TCP 压测都经过网线；USB 用于 PC 管理和 UI。':'Physical cable loop: {host} ⇄ {peer} · 10 GbE · MTU {mtu}. Inputs, results and bidirectional TCP load all cross the cable; USB serves PC management and UI.',
};
let language;
try {language=localStorage.getItem('thor-language') || 'zh';} catch {language='zh';}
if (!['zh','en'].includes(language)) language='zh';
const bindings=[];
export function t(key, values={}) {
  let value=language==='en' ? translations[key] ?? key : key;
  for(const [name,replacement] of Object.entries(values)) value=value.replaceAll(`{${name}}`,String(replacement));
  return value;
}
export function className(label) {
  const names={person:'行人',bicycle:'自行车',car:'车辆',motorcycle:'摩托车',bus:'巴士',truck:'卡车','traffic light':'信号灯','stop sign':'停车标志'};
  return language==='en' ? label : names[label] || label;
}
export function bindTranslations() {
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  while(walker.nextNode()) {
    const node=walker.currentNode,key=node.textContent.trim();
    if(Object.hasOwn(translations,key)) bindings.push({node,key,prefix:node.textContent.match(/^\s*/)[0],suffix:node.textContent.match(/\s*$/)[0]});
  }
  for(const node of document.querySelectorAll('[title],[aria-label],[alt]')) {
    for(const attribute of ['title','aria-label','alt']) {
      const key=node.getAttribute(attribute);
      if(Object.hasOwn(translations,key)) bindings.push({node,key,attribute});
    }
  }
}
export function setLanguage(next) {
  language=next;
  try {localStorage.setItem('thor-language',language);} catch {}
  document.documentElement.lang=language==='en'?'en':'zh-CN';
  for(const {node,key,attribute,prefix,suffix} of bindings) {
    if(!node.isConnected)continue;
    if(attribute)node.setAttribute(attribute,t(key));
    else node.textContent=prefix+t(key)+suffix;
  }
  document.getElementById('language-select').value=language;
}
export function currentLanguage() {return language;}
