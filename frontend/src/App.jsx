import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);

  const [detections, setDetections] = useState([]);
  const [totalObjects, setTotalObjects] = useState(0);
  const [inferenceTime, setInferenceTime] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setSelectedFile(file);
    setOriginalImage(URL.createObjectURL(file));

    // Reset previous results
    setResultImage(null);
    setDetections([]);
    setTotalObjects(0);
    setInferenceTime(null);
    setError("");
  };


  const runDetection = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();

    formData.append(
      "file",
      selectedFile
    );

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/detect",
        formData
      );

      const data = response.data;

      setResultImage(
        `data:image/jpeg;base64,${data.image}`
      );

      setDetections(data.detections);
      setTotalObjects(data.total_objects);
      setInferenceTime(data.inference_time_ms);

    } catch (err) {
      console.error(err);

      setError(
        "Detection failed. Make sure the backend is running."
      );
    }

    setLoading(false);
  };


  return (
    <div className="app">

      <header>
        <div>
          <h1>Road Object Detection System</h1>

          <p>
            YOLOv5 Model Testing Dashboard
          </p>
        </div>

        <div className="status">
          ● System Ready
        </div>
      </header>


      <main>

        {/* Upload Section */}

        <section className="control-panel">

          <div className="upload-box">

            <label>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
              />

              <span className="upload-icon">
                📁
              </span>

              <span>
                Upload Road Image
              </span>

              <small>
                JPG, JPEG, PNG
              </small>

            </label>

          </div>


          <button
            className="detect-button"
            onClick={runDetection}
            disabled={!selectedFile || loading}
          >

            {loading
              ? "Running Detection..."
              : "Run Detection"
            }

          </button>

        </section>


        {error && (
          <p className="error">
            {error}
          </p>
        )}


        {/* Statistics */}

        <section className="statistics">

          <div className="stat-card">

            <span>Total Objects</span>

            <strong>
              {totalObjects}
            </strong>

          </div>


          <div className="stat-card">

            <span>Inference Time</span>

            <strong>
              {inferenceTime
                ? `${inferenceTime} ms`
                : "--"
              }
            </strong>

          </div>


          <div className="stat-card">

            <span>Detection Status</span>

            <strong>
              {resultImage
                ? "Completed"
                : "Waiting"
              }
            </strong>

          </div>

        </section>


        {/* Images */}

        <section className="results">

          <div className="image-card">

            <h2>Original Image</h2>

            {originalImage ? (

              <img
                src={originalImage}
                alt="Original"
              />

            ) : (

              <div className="placeholder">
                Upload an image to begin
              </div>

            )}

          </div>


          <div className="image-card">

            <h2>Detection Result</h2>

            {resultImage ? (

              <img
                src={resultImage}
                alt="Detection Result"
              />

            ) : (

              <div className="placeholder">

                {loading
                  ? "YOLOv5 is analyzing the image..."
                  : "Detection result will appear here"
                }

              </div>

            )}

          </div>

        </section>


        {/* Detection Table */}

        {detections.length > 0 && (

          <section className="detection-panel">

            <h2>
              Detected Objects
            </h2>


            <table>

              <thead>

                <tr>
                  <th>#</th>
                  <th>Object Class</th>
                  <th>Confidence</th>
                </tr>

              </thead>


              <tbody>

                {detections.map(
                  (detection, index) => (

                    <tr key={index}>

                      <td>
                        {index + 1}
                      </td>

                      <td>
                        {detection.class}
                      </td>

                      <td>
                        {detection.confidence}%
                      </td>

                    </tr>

                  )
                )}

              </tbody>

            </table>

          </section>

        )}

      </main>

    </div>
  );
}

export default App;