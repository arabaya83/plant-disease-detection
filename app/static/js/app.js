const videoEl = document.getElementById("camera");
const canvasEl = document.getElementById("snapshot");
const previewEl = document.getElementById("preview");
const fileInputEl = document.getElementById("fileInput");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingEl = document.getElementById("loading");
const resultEl = document.getElementById("result");

let currentBlob = null;

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    videoEl.srcObject = stream;
  } catch (err) {
    console.error(err);
    alert("Camera unavailable. Please use upload fallback.");
  }
}

function captureFrame() {
  const width = videoEl.videoWidth || 1280;
  const height = videoEl.videoHeight || 720;
  canvasEl.width = width;
  canvasEl.height = height;
  const ctx = canvasEl.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, width, height);

  // Browser-side lightweight resize/compression
  const maxSide = 1280;
  let tw = width;
  let th = height;
  if (Math.max(width, height) > maxSide) {
    const scale = maxSide / Math.max(width, height);
    tw = Math.round(width * scale);
    th = Math.round(height * scale);
  }

  const tmp = document.createElement("canvas");
  tmp.width = tw;
  tmp.height = th;
  tmp.getContext("2d").drawImage(canvasEl, 0, 0, tw, th);

  tmp.toBlob((blob) => {
    currentBlob = blob;
    previewEl.src = URL.createObjectURL(blob);
    previewEl.classList.remove("hidden");
  }, "image/jpeg", 0.85);
}

function onUploadChange(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  currentBlob = file;
  previewEl.src = URL.createObjectURL(file);
  previewEl.classList.remove("hidden");
}

function renderResult(data) {
  resultEl.innerHTML = "";
  const title = document.createElement("h2");
  title.textContent = data.message;
  resultEl.appendChild(title);

  if (!data.results || data.results.length === 0) {
    const p = document.createElement("p");
    p.textContent = "Try a clearer image with better leaf visibility.";
    resultEl.appendChild(p);
    return;
  }

  data.results.forEach((item) => {
    const card = document.createElement("div");
    card.className = "result-item";
    const healthCls = item.healthy_or_diseased === "Healthy" ? "healthy" : "diseased";

    card.innerHTML = `
      <h3>Leaf #${item.leaf_id}</h3>
      <p><strong>Crop:</strong> ${item.crop_name}</p>
      <p><strong>Disease:</strong> ${item.disease_name}</p>
      <p><strong>Confidence:</strong> ${(item.confidence * 100).toFixed(1)}%</p>
      <span class="badge ${healthCls}">${item.healthy_or_diseased}</span>
      <p>${item.short_description}</p>
    `;

    if (item.heatmap_path) {
      const img = document.createElement("img");
      img.className = "heatmap";
      img.alt = `Grad-CAM leaf ${item.leaf_id}`;
      img.src = item.heatmap_path;
      card.appendChild(img);
    }

    resultEl.appendChild(card);
  });
}

async function analyzeImage() {
  if (!currentBlob) {
    alert("Capture or upload an image first.");
    return;
  }

  loadingEl.classList.remove("hidden");
  resultEl.innerHTML = "";

  const formData = new FormData();
  formData.append("file", currentBlob, "leaf.jpg");

  try {
    const res = await fetch("/infer", { method: "POST", body: formData });
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    console.error(err);
    resultEl.textContent = "Failed to analyze image.";
  } finally {
    loadingEl.classList.add("hidden");
  }
}

startCameraBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", captureFrame);
fileInputEl.addEventListener("change", onUploadChange);
analyzeBtn.addEventListener("click", analyzeImage);
