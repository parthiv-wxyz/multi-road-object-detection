import { useEffect, useRef, useState } from "react";
import axios from "axios";
import sampleRoad from "./assets/sample-road.jpg";
import "./App.css";

const ZOOM_STEP = 0.25;
const MIN_ZOOM = 1;
const MAX_ZOOM = 4;

function Icon({ children, label }) {
  return <span aria-hidden="true" className="icon" title={label}>{children}</span>;
}

function ZoomToolbar({ zoom, pan, onZoom, onReset, onFullscreen, showFullscreen }) {
  const stopViewerDrag = (event) => event.stopPropagation();

  return (
    <div className="zoom-toolbar" aria-label="Image zoom controls" onPointerDown={stopViewerDrag} onWheel={stopViewerDrag}>
      <button type="button" onClick={() => onZoom(zoom - ZOOM_STEP)} disabled={zoom === MIN_ZOOM} aria-label="Zoom out" title="Zoom out"><Icon label="Zoom out">&#8722;</Icon></button>
      <output aria-label={`Zoom level ${Math.round(zoom * 100)} percent`}>{Math.round(zoom * 100)}%</output>
      <button type="button" onClick={() => onZoom(zoom + ZOOM_STEP)} disabled={zoom === MAX_ZOOM} aria-label="Zoom in" title="Zoom in"><Icon label="Zoom in">+</Icon></button>
      <span className="toolbar-divider" />
      <button type="button" onClick={onReset} disabled={zoom === MIN_ZOOM && pan.x === 0 && pan.y === 0} aria-label="Reset image view" title="Reset image view"><Icon label="Reset image view">&#8634;</Icon></button>
      {showFullscreen && <button type="button" onClick={onFullscreen} aria-label="Open fullscreen image" title="Open fullscreen image"><Icon label="Open fullscreen image">&#8599;</Icon></button>}
    </div>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [totalObjects, setTotalObjects] = useState(0);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const dragStart = useRef(null);

  useEffect(() => () => {
    if (originalImage) URL.revokeObjectURL(originalImage);
  }, [originalImage]);

  const resetViewer = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  useEffect(() => {
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        resetViewer();
        setIsFullscreen(false);
      }
    };

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  const updateZoom = (nextZoom) => {
    const clampedZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom));
    setZoom(clampedZoom);
    if (clampedZoom === MIN_ZOOM) setPan({ x: 0, y: 0 });
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setOriginalImage(URL.createObjectURL(file));
    setResultImage(null);
    setDetections([]);
    setTotalObjects(0);
    setInferenceTime(null);
    setError("");
    resetViewer();
  };

  const runDetection = async () => {
    if (!selectedFile) {
      setError("Choose a road image before running detection.");
      return;
    }
    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const { data } = await axios.post("http://127.0.0.1:8000/detect", formData);
      setResultImage(`data:image/jpeg;base64,${data.image}`);
      setDetections(data.detections);
      setTotalObjects(data.total_objects);
      setInferenceTime(data.inference_time_ms);
      resetViewer();
    } catch (requestError) {
      console.error(requestError);
      setError("Detection failed. Check that the local model service is running.");
    } finally {
      setLoading(false);
    }
  };

  const handlePointerDown = (event) => {
    if (zoom === MIN_ZOOM || event.target.tagName !== "IMG") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStart.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
  };

  const handlePointerMove = (event) => {
    if (!dragStart.current) return;
    setPan({ x: dragStart.current.panX + event.clientX - dragStart.current.x, y: dragStart.current.panY + event.clientY - dragStart.current.y });
  };

  const handlePointerUp = () => { dragStart.current = null; };
  const handleViewerWheel = (event) => {
    event.preventDefault();
    updateZoom(zoom + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP));
  };
  const toggleZoom = () => updateZoom(zoom === MIN_ZOOM ? 2 : MIN_ZOOM);
  const openFullscreen = () => {
    resetViewer();
    setIsFullscreen(true);
  };
  const closeFullscreen = () => {
    resetViewer();
    setIsFullscreen(false);
  };
  const resultSource = resultImage || originalImage || sampleRoad;
  const resultAlt = resultImage ? "Road detection result" : "Road image preview";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#workspace" aria-label="RoadSight dashboard"><span className="brand-mark"><span /></span><span>RoadSight</span></a>
        <div className="topbar-meta"><span className="model-name">IDD Final</span><span className="ready-status"><i /> Model ready</span></div>
      </header>
      <main id="workspace" className="workspace">
        <section className="workspace-heading" aria-labelledby="page-title">
          <div><p className="eyebrow">Detection workspace</p><h1 id="page-title">Inspect every road scene with clarity.</h1><p className="heading-copy">Upload an image, run the model, then inspect every detection at the level of detail you need.</p></div>
          <label className="file-action"><input type="file" accept="image/png,image/jpeg,image/jpg" onChange={handleFileChange} /><Icon label="Upload image">&#8593;</Icon><span>{selectedFile ? "Replace image" : "Choose image"}</span></label>
        </section>
        <section className="control-bar" aria-label="Detection controls">
          <div className="selected-file"><span className="file-icon"><Icon label="Image">&#9635;</Icon></span><div><span className="control-label">Input image</span><strong>{selectedFile?.name || "No image selected"}</strong></div></div>
          <button className="run-button" onClick={runDetection} disabled={!selectedFile || loading}><Icon label="Run detection">&#9654;</Icon><span>{loading ? "Analyzing scene" : "Run detection"}</span></button>
        </section>
        {error && <p className="error-message" role="alert">{error}</p>}
        <section className="metrics" aria-label="Detection summary">
          <article className="metric"><span>Objects found</span><strong>{totalObjects}</strong><small>{resultImage ? "Across this frame" : "Waiting for a result"}</small></article>
          <article className="metric"><span>Inference time</span><strong>{inferenceTime ? `${inferenceTime} ms` : "--"}</strong><small>{inferenceTime ? "Model response" : "No measurement yet"}</small></article>
          <article className="metric metric-status"><span>Run status</span><strong>{loading ? "Analyzing" : resultImage ? "Complete" : "Standby"}</strong><small><i /> {loading ? "Model is working" : resultImage ? "Result is ready" : "Ready when you are"}</small></article>
        </section>
        <section className="inspection-grid" aria-label="Image comparison">
          <article className="image-panel"><div className="panel-header"><div><p className="panel-kicker">Source frame</p><h2>Original image</h2></div>{originalImage && <span className="image-badge">Uploaded</span>}</div><div className="source-canvas"><img src={originalImage || sampleRoad} alt={originalImage ? "Uploaded road scene" : "Sample road scene"} />{!originalImage && <span className="sample-note">Sample road frame</span>}</div></article>
          <article className="image-panel result-panel">
            <div className="panel-header"><div><p className="panel-kicker">Model output</p><h2>Detection result</h2></div><span className={resultImage ? "image-badge result-ready" : "image-badge"}>{resultImage ? "Complete" : "Preview"}</span></div>
            <div className={`zoom-canvas ${zoom > MIN_ZOOM ? "is-zoomed" : ""}`} onWheel={handleViewerWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onPointerLeave={handlePointerUp} onPointerCancel={handlePointerUp} onDoubleClick={toggleZoom}>
              <img src={resultSource} alt={resultAlt} draggable="false" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }} />
              {!resultImage && <span className="sample-note">Run a detection to see annotations</span>}
              <ZoomToolbar zoom={zoom} pan={pan} onZoom={updateZoom} onReset={resetViewer} onFullscreen={openFullscreen} showFullscreen />
            </div>
            <p className="viewer-hint">Use the controls or mouse wheel to zoom. Drag to pan when zoomed.</p>
          </article>
        </section>
        <section className="detection-list" aria-labelledby="detections-heading">
          <div className="list-heading"><div><p className="panel-kicker">Detection log</p><h2 id="detections-heading">Detected objects</h2></div><span>{detections.length} items</span></div>
          {detections.length > 0 ? <div className="table-wrap"><table><thead><tr><th>Object</th><th>Confidence</th><th>Location</th></tr></thead><tbody>{detections.map((detection, index) => <tr key={`${detection.class}-${index}`}><td><span className="object-index">{String(index + 1).padStart(2, "0")}</span>{detection.class}</td><td><span className="confidence">{detection.confidence}%</span></td><td>{Math.round(detection.xmin)}, {Math.round(detection.ymin)} to {Math.round(detection.xmax)}, {Math.round(detection.ymax)}</td></tr>)}</tbody></table></div> : <div className="empty-log"><span><Icon label="No detections yet">&#9678;</Icon></span><p>Detected objects will appear here after a completed run.</p></div>}
        </section>
      </main>
      {isFullscreen && <div className="fullscreen-backdrop" role="dialog" aria-modal="true" aria-label="Fullscreen detection result" onClick={closeFullscreen}><div className="fullscreen-view" onClick={(event) => event.stopPropagation()}><div className="fullscreen-header"><span>Detection result</span><button onClick={closeFullscreen} aria-label="Close fullscreen image" title="Close"><Icon label="Close">&#215;</Icon></button></div><div className={`fullscreen-canvas ${zoom > MIN_ZOOM ? "is-zoomed" : ""}`} onWheel={handleViewerWheel} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onPointerLeave={handlePointerUp} onPointerCancel={handlePointerUp} onDoubleClick={toggleZoom}><img src={resultSource} alt={resultAlt} draggable="false" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }} /><ZoomToolbar zoom={zoom} pan={pan} onZoom={updateZoom} onReset={resetViewer} showFullscreen={false} /></div></div></div>}
    </div>
  );
}

export default App;
