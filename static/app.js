const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const preview = document.getElementById("preview");
const hint = document.getElementById("hint");
const result = document.getElementById("result");
const label = document.getElementById("label");
const barFill = document.getElementById("bar-fill");

async function classify(file) {
  if (!file) return;
  URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  hint.hidden = true;
  result.hidden = false;
  result.className = "";
  label.textContent = "Thinking…";
  barFill.style.width = "0";

  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/predict", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    const percent = (data.confidence * 100).toFixed(1);
    label.textContent = `${data.label === "cat" ? "🐱" : "🐶"} It's a ${data.label} (${percent}% sure)`;
    barFill.style.width = `${percent}%`;
  } catch (err) {
    result.className = "error";
    label.textContent = `Error: ${err.message}`;
  }
}

fileInput.addEventListener("change", () => classify(fileInput.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  classify(e.dataTransfer.files[0]);
});

fetch("/health")
  .then((r) => r.json())
  .then((d) => { document.getElementById("model").textContent = `Model: ${d.model}`; })
  .catch(() => {});
