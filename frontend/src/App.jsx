import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const featureCards = [
  {
    eyebrow: "Fast upload",
    title: "Drop a road scene and start instantly",
    description:
      "Choose a frame from CCTV, dashcam, or field capture and send it to the detector in one click.",
  },
  {
    eyebrow: "Clear output",
    title: "Review original and annotated frames side by side",
    description:
      "Compare raw input with the processed result without losing the visual context of the scene.",
  },
  {
    eyebrow: "Built for testing",
    title: "A focused interface for model validation",
    description:
      "Designed for quick iteration while you verify road object predictions from your YOLO backend.",
  },
];

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (originalImage) URL.revokeObjectURL(originalImage);
      if (resultImage) URL.revokeObjectURL(resultImage);
    };
  }, [originalImage, resultImage]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (originalImage) URL.revokeObjectURL(originalImage);
    if (resultImage) URL.revokeObjectURL(resultImage);

    setSelectedFile(file);
    setOriginalImage(URL.createObjectURL(file));
    setResultImage(null);
    setError("");
  };

  const runDetection = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");

    if (resultImage) {
      URL.revokeObjectURL(resultImage);
      setResultImage(null);
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/detect",
        formData,
        { responseType: "blob" }
      );

      setResultImage(URL.createObjectURL(response.data));
    } catch (err) {
      console.error(err);
      setError("Detection failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <main className="app">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">Computer Vision Dashboard</span>
            <h1>Modern road object detection UI for your YOLO workflow.</h1>
            <p className="hero-text">
              Upload an image, run inference, and inspect the annotated output in
              a cleaner testing environment built for fast model evaluation.
            </p>

            <div className="hero-actions">
              <label className="primary-upload">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
                <span>{selectedFile ? "Replace image" : "Choose image"}</span>
              </label>

              <button
                className="detect-button"
                onClick={runDetection}
                disabled={!selectedFile || loading}
              >
                {loading ? "Running detection..." : "Run detection"}
              </button>
            </div>

            <div className="status-strip">
              <div className="status-card">
                <span className="status-label">Selected file</span>
                <strong>{selectedFile ? selectedFile.name : "No file yet"}</strong>
              </div>
              <div className="status-card">
                <span className="status-label">Pipeline status</span>
                <strong>{loading ? "Processing image" : "Ready for inference"}</strong>
              </div>
            </div>

            {error && <p className="error-banner">{error}</p>}
          </div>

          <div className="hero-panel">
            <div className="panel-badge">YOLOv5 testing</div>
            <h2>Model run overview</h2>
            <p>
              Keep the interaction simple while presenting the workflow in a more
              polished, dashboard-like layout.
            </p>

            <div className="metric-grid">
              <div className="metric-card">
                <span>Input</span>
                <strong>Road image</strong>
              </div>
              <div className="metric-card">
                <span>Output</span>
                <strong>Annotated frame</strong>
              </div>
              <div className="metric-card">
                <span>Endpoint</span>
                <strong>`/detect`</strong>
              </div>
              <div className="metric-card">
                <span>Mode</span>
                <strong>Single image</strong>
              </div>
            </div>
          </div>
        </section>

        <section className="feature-grid">
          {featureCards.map((card) => (
            <article className="feature-card" key={card.title}>
              <span className="feature-eyebrow">{card.eyebrow}</span>
              <h3>{card.title}</h3>
              <p>{card.description}</p>
            </article>
          ))}
        </section>

        <section className="results-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Visual comparison</span>
              <h2>Before and after detection</h2>
            </div>
            <p>
              Review your uploaded frame alongside the processed output without
              leaving the page.
            </p>
          </div>

          <div className="results-grid">
            <article className="image-card">
              <div className="card-header">
                <div>
                  <span className="card-label">Source</span>
                  <h3>Original image</h3>
                </div>
              </div>

              {originalImage ? (
                <img src={originalImage} alt="Original upload preview" />
              ) : (
                <div className="placeholder">
                  <p>Upload an image to preview the source frame here.</p>
                </div>
              )}
            </article>

            <article className="image-card">
              <div className="card-header">
                <div>
                  <span className="card-label">Result</span>
                  <h3>Detection output</h3>
                </div>
              </div>

              {resultImage ? (
                <img src={resultImage} alt="Detection result preview" />
              ) : (
                <div className="placeholder">
                  <p>
                    {loading
                      ? "Inference is running. Your annotated image will appear here."
                      : "Run detection to view the annotated output."}
                  </p>
                </div>
              )}
            </article>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
