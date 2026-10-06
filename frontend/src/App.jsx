import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  Aperture,
  ArrowUp,
  ClipboardPaste,
  FileImage,
  Image as ImageIcon,
  Maximize2,
  Play,
  RefreshCcw,
  ScanSearch,
  Timer,
  UploadCloud,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import sampleRoad from "./assets/sample-road.jpg";
import "./App.css";

const ZOOM_STEP = 0.25;
const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg"];

const hasFiles = (event) =>
  Array.from(event.dataTransfer?.types || []).includes("Files");

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0 },
};

const trailAnimations = [
  { x: ["-18vw", "18vw", "-8vw"], y: ["6vh", "24vh", "8vh"], duration: 18 },
  { x: ["14vw", "-16vw", "10vw"], y: ["40vh", "22vh", "46vh"], duration: 22 },
  { x: ["-10vw", "16vw", "-14vw"], y: ["70vh", "54vh", "72vh"], duration: 20 },
];

function MotionBackdrop() {
  return (
    <div className="live-backdrop" aria-hidden="true">
      <div className="backdrop-grid" />
      {trailAnimations.map((trail, index) => (
        <motion.span
          className={`motion-trail trail-${index + 1}`}
          key={trail.duration}
          animate={{ x: trail.x, y: trail.y, rotate: [0, 8, -6, 0] }}
          transition={{
            duration: trail.duration,
            ease: "easeInOut",
            repeat: Infinity,
            repeatType: "mirror",
          }}
        />
      ))}
      <motion.span
        className="scanner-line"
        animate={{ x: ["-25vw", "125vw"] }}
        transition={{ duration: 12, ease: "linear", repeat: Infinity }}
      />
    </div>
  );
}

function IconButton({ children, label, ...props }) {
  return (
    <button type="button" aria-label={label} title={label} {...props}>
      {children}
    </button>
  );
}

function ZoomToolbar({
  zoom,
  pan,
  onZoom,
  onReset,
  onFullscreen,
  showFullscreen,
}) {
  const stopViewerDrag = (event) => event.stopPropagation();

  return (
    <motion.div
      className="zoom-toolbar"
      aria-label="Image zoom controls"
      onPointerDown={stopViewerDrag}
      onWheel={stopViewerDrag}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.24 }}
    >
      <IconButton
        onClick={() => onZoom(zoom - ZOOM_STEP)}
        disabled={zoom === MIN_ZOOM}
        label="Zoom out"
      >
        <ZoomOut size={16} />
      </IconButton>
      <output aria-label={`Zoom level ${Math.round(zoom * 100)} percent`}>
        {Math.round(zoom * 100)}%
      </output>
      <IconButton
        onClick={() => onZoom(zoom + ZOOM_STEP)}
        disabled={zoom === MAX_ZOOM}
        label="Zoom in"
      >
        <ZoomIn size={16} />
      </IconButton>
      <span className="toolbar-divider" />
      <IconButton
        onClick={onReset}
        disabled={zoom === MIN_ZOOM && pan.x === 0 && pan.y === 0}
        label="Reset image view"
      >
        <RefreshCcw size={15} />
      </IconButton>
      {showFullscreen && (
        <IconButton onClick={onFullscreen} label="Open fullscreen image">
          <Maximize2 size={15} />
        </IconButton>
      )}
    </motion.div>
  );
}

function MetricCard({ icon: Icon, label, value, detail, delay = 0 }) {
  return (
    <motion.article
      className="metric"
      variants={fadeUp}
      initial="hidden"
      animate="visible"
      transition={{ duration: 0.45, delay }}
      whileHover={{ y: -4 }}
    >
      <span className="metric-icon">
        <Icon size={18} />
      </span>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </motion.article>
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
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef(null);
  const dragDepth = useRef(0);

  useEffect(
    () => () => {
      if (originalImage) URL.revokeObjectURL(originalImage);
    },
    [originalImage],
  );

  const resetViewer = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        resetViewer();
        setIsFullscreen(false);
      }
    };

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [resetViewer]);

  const updateZoom = (nextZoom) => {
    const clampedZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom));
    setZoom(clampedZoom);
    if (clampedZoom === MIN_ZOOM) setPan({ x: 0, y: 0 });
  };

  // Single entry point for every input method (picker, drag and drop, paste).
  // The previous object URL is revoked by the effect above when it changes.
  const loadFile = useCallback(
    (file) => {
      if (!file) return;
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Unsupported file. Use a PNG or JPEG road image.");
        return;
      }
      setSelectedFile(file);
      setOriginalImage(URL.createObjectURL(file));
      setResultImage(null);
      setDetections([]);
      setTotalObjects(0);
      setInferenceTime(null);
      setError("");
      resetViewer();
    },
    [resetViewer],
  );

  const handleFileChange = (event) => {
    loadFile(event.target.files?.[0]);
    event.target.value = ""; // allow re-selecting the same file
  };

  // Page-wide drag and drop + clipboard paste
  useEffect(() => {
    const onDragEnter = (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth.current += 1;
      setIsDragging(true);
    };
    const onDragOver = (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault(); // required so the drop event fires
      event.dataTransfer.dropEffect = "copy";
    };
    const onDragLeave = (event) => {
      if (!hasFiles(event)) return;
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setIsDragging(false);
    };
    const onDrop = (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      dragDepth.current = 0;
      setIsDragging(false);
      const file = Array.from(event.dataTransfer.files).find((f) =>
        f.type.startsWith("image/"),
      );
      if (file) loadFile(file);
      else setError("No image found in the dropped item.");
    };
    const onPaste = (event) => {
      const item = Array.from(event.clipboardData?.items || []).find(
        (i) => i.kind === "file" && i.type.startsWith("image/"),
      );
      if (!item) return;
      event.preventDefault();
      loadFile(item.getAsFile());
    };

    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    window.addEventListener("paste", onPaste);
    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
      window.removeEventListener("paste", onPaste);
    };
  }, [loadFile]);

  const runDetection = async () => {
    if (!selectedFile) {
      setError("Add a road image (choose, drag and drop, or paste) before running detection.");
      return;
    }
    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const { data } = await axios.post(
        "http://127.0.0.1:8000/detect",
        formData,
      );
      setResultImage(`data:image/jpeg;base64,${data.image}`);
      setDetections(data.detections);
      setTotalObjects(data.total_objects);
      setInferenceTime(data.inference_time_ms);
      resetViewer();
      requestAnimationFrame(() => {
        document
          .getElementById("results")
          ?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } catch (requestError) {
      console.error(requestError);
      setError(
        "Detection failed. Check that the local model service is running.",
      );
    } finally {
      setLoading(false);
    }
  };

  const handlePointerDown = (event) => {
    if (zoom === MIN_ZOOM || event.target.tagName !== "IMG") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStart.current = {
      x: event.clientX,
      y: event.clientY,
      panX: pan.x,
      panY: pan.y,
    };
  };

  const handlePointerMove = (event) => {
    if (!dragStart.current) return;
    setPan({
      x: dragStart.current.panX + event.clientX - dragStart.current.x,
      y: dragStart.current.panY + event.clientY - dragStart.current.y,
    });
  };

  const handlePointerUp = () => {
    dragStart.current = null;
  };
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
  const resultAlt = resultImage
    ? "Road detection result"
    : "Road image preview";

  return (
    <div className="app-shell">
      <MotionBackdrop />
      <AnimatePresence>
        {isDragging && (
          <motion.div
            className="drop-overlay"
            aria-hidden="true"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            <div className="drop-overlay-card">
              <UploadCloud size={40} />
              <strong>Drop image to load it</strong>
              <span>PNG or JPEG</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <header className="topbar">
        <a className="brand" href="#workspace" aria-label="RoadSight dashboard">
          <span className="brand-mark">
            <Aperture size={17} />
          </span>
          <span>RoadSight</span>
        </a>
        <nav className="topbar-links" aria-label="Dashboard sections">
          <a href="#workspace">Workspace</a>
          <a href="#results">Results</a>
          <a href="#detections">Log</a>
        </nav>
        <div className="topbar-meta">
          <span className="model-name">IDD Final</span>
          <span className="ready-status">
            <i /> Model ready
          </span>
        </div>
      </header>

      <main id="workspace" className="workspace">
        <motion.section
          className="hero"
          aria-labelledby="page-title"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: "easeOut" }}
        >
          <div className="hero-copy">
            <p className="eyebrow">Detection workspace</p>
            <h1 id="page-title">Road scene intelligence, tuned for review.</h1>
            <p className="heading-copy">
              Upload a frame, run the local YOLO model, and inspect every
              detection with responsive zoom controls and a clean audit trail.
            </p>
            <div className="hero-actions">
              <label className="file-action">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleFileChange}
                />
                <ArrowUp size={17} />
                <span>{selectedFile ? "Replace image" : "Choose image"}</span>
              </label>
              <button
                className="run-button"
                onClick={runDetection}
                disabled={!selectedFile || loading}
              >
                <Play size={17} fill="currentColor" />
                <span>{loading ? "Analyzing scene" : "Run detection"}</span>
              </button>
            </div>
            <p className="input-hint">
              <ClipboardPaste size={14} />
              <span>
                or drag and drop an image anywhere, or paste with{" "}
                <kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>V</kbd>
              </span>
            </p>
          </div>
          <motion.div
            className="hero-preview"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.14 }}
          >
            <img
              src={originalImage || sampleRoad}
              alt={originalImage ? "Uploaded road scene" : "Sample road scene"}
            />
            <span className="preview-chip">
              <ScanSearch size={14} />
              {resultImage ? "Detection complete" : "Ready to inspect"}
            </span>
          </motion.div>
        </motion.section>

        <motion.section
          className="control-bar"
          aria-label="Detection controls"
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.45, delay: 0.1 }}
        >
          <div className="selected-file">
            <span className="file-icon">
              <FileImage size={20} />
            </span>
            <div>
              <span className="control-label">Input image</span>
              <strong>{selectedFile?.name || "No image selected"}</strong>
            </div>
          </div>
          <a className="jump-link" href="#results">
            View results
          </a>
        </motion.section>

        <AnimatePresence>
          {error && (
            <motion.p
              className="error-message"
              role="alert"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
            >
              {error}
            </motion.p>
          )}
        </AnimatePresence>

        <section className="metrics" aria-label="Detection summary">
          <MetricCard
            icon={ScanSearch}
            label="Objects found"
            value={totalObjects}
            detail={resultImage ? "Across this frame" : "Waiting for a result"}
            delay={0.16}
          />
          <MetricCard
            icon={Timer}
            label="Inference time"
            value={inferenceTime ? `${inferenceTime} ms` : "--"}
            detail={inferenceTime ? "Model response" : "No measurement yet"}
            delay={0.22}
          />
          <MetricCard
            icon={Activity}
            label="Run status"
            value={loading ? "Analyzing" : resultImage ? "Complete" : "Standby"}
            detail={
              loading
                ? "Model is working"
                : resultImage
                  ? "Result is ready"
                  : "Ready when you are"
            }
            delay={0.28}
          />
        </section>

        <section
          id="results"
          className="inspection-grid"
          aria-label="Image comparison"
        >
          <motion.article
            className="image-panel"
            variants={fadeUp}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.45 }}
          >
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Source frame</p>
                <h2>Original image</h2>
              </div>
              {originalImage && <span className="image-badge">Uploaded</span>}
            </div>
            <div className="source-canvas">
              <img
                src={originalImage || sampleRoad}
                alt={originalImage ? "Uploaded road scene" : "Sample road scene"}
              />
              {!originalImage && (
                <span className="sample-note">Sample road frame</span>
              )}
            </div>
          </motion.article>

          <motion.article
            className="image-panel result-panel"
            variants={fadeUp}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.45, delay: 0.08 }}
          >
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Model output</p>
                <h2>Detection result</h2>
              </div>
              <span
                className={
                  resultImage ? "image-badge result-ready" : "image-badge"
                }
              >
                {resultImage ? "Complete" : "Preview"}
              </span>
            </div>
            <div
              className={`zoom-canvas ${zoom > MIN_ZOOM ? "is-zoomed" : ""}`}
              onWheel={handleViewerWheel}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
              onPointerCancel={handlePointerUp}
              onDoubleClick={toggleZoom}
            >
              <motion.img
                src={resultSource}
                alt={resultAlt}
                draggable="false"
                style={{
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                }}
                key={resultSource}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.28 }}
              />
              {!resultImage && (
                <span className="sample-note">
                  Run a detection to see annotations
                </span>
              )}
              <ZoomToolbar
                zoom={zoom}
                pan={pan}
                onZoom={updateZoom}
                onReset={resetViewer}
                onFullscreen={openFullscreen}
                showFullscreen
              />
            </div>
            <p className="viewer-hint">
              Mouse wheel zooms, double-click toggles zoom, and dragging pans the frame.
            </p>
          </motion.article>
        </section>

        <motion.section
          id="detections"
          className="detection-list"
          aria-labelledby="detections-heading"
          variants={fadeUp}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.16 }}
          transition={{ duration: 0.45 }}
        >
          <div className="list-heading">
            <div>
              <p className="panel-kicker">Detection log</p>
              <h2 id="detections-heading">Detected objects</h2>
            </div>
            <span>
              {detections.length > 0
                ? `${new Set(detections.map((d) => d.class)).size} classes`
                : "0 items"}
            </span>
          </div>
          <AnimatePresence mode="wait">
            {detections.length > 0 ? (
              <motion.div
                className="table-wrap"
                key="detections"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <table>
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>Count</th>
                      <th>Avg Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      detections.reduce((acc, d) => {
                        if (!acc[d.class]) acc[d.class] = { count: 0, totalConf: 0 };
                        acc[d.class].count += 1;
                        acc[d.class].totalConf += d.confidence;
                        return acc;
                      }, {}),
                    )
                      .sort(([, a], [, b]) => b.count - a.count)
                      .map(([cls, { count, totalConf }], index) => (
                        <motion.tr
                          key={cls}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.28, delay: index * 0.03 }}
                        >
                          <td>
                            <span className="object-index">
                              {String(index + 1).padStart(2, "0")}
                            </span>
                            {cls}
                          </td>
                          <td>{count}</td>
                          <td>
                            <span className="confidence">
                              {(totalConf / count).toFixed(1)}%
                            </span>
                          </td>
                        </motion.tr>
                      ))}
                  </tbody>
                </table>
              </motion.div>
            ) : (
              <motion.div
                className="empty-log"
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <span>
                  <ImageIcon size={18} />
                </span>
                <p>Detected objects will appear here after a completed run.</p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.section>
      </main>

      <AnimatePresence>
        {isFullscreen && (
          <motion.div
            className="fullscreen-backdrop"
            role="dialog"
            aria-modal="true"
            aria-label="Fullscreen detection result"
            onClick={closeFullscreen}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="fullscreen-view"
              onClick={(event) => event.stopPropagation()}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.22 }}
            >
              <div className="fullscreen-header">
                <span>Detection result</span>
                <IconButton
                  onClick={closeFullscreen}
                  label="Close fullscreen image"
                >
                  <X size={19} />
                </IconButton>
              </div>
              <div
                className={`fullscreen-canvas ${zoom > MIN_ZOOM ? "is-zoomed" : ""}`}
                onWheel={handleViewerWheel}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
                onPointerCancel={handlePointerUp}
                onDoubleClick={toggleZoom}
              >
                <img
                  src={resultSource}
                  alt={resultAlt}
                  draggable="false"
                  style={{
                    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  }}
                />
                <ZoomToolbar
                  zoom={zoom}
                  pan={pan}
                  onZoom={updateZoom}
                  onReset={resetViewer}
                  showFullscreen={false}
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;