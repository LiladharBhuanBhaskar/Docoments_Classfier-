function $(id) {
  return document.getElementById(id);
}

let lastDocumentJson = null;

function setLastDocumentJson(value) {
  lastDocumentJson = value;
  window.dispatchEvent(new CustomEvent("document-json", { detail: value }));
}

function getLastDocumentJson() {
  return lastDocumentJson;
}

const DOC_STORAGE_KEY = "doc_parser.selected_doc";

function normalizeDocId(value) {
  const raw = String(value || "")
    .trim()
    .toLowerCase();
  if (!raw) return null;
  const v = raw.replace(/[\s-]+/g, "_");
  if (v === "pan") return "pan";
  if (v === "aadhaar" || v === "aadhar") return "aadhaar";
  if (v === "voter_id" || v === "voterid" || v === "voter" || v === "voters") return "voter_id";
  if (v === "driving_license" || v === "driving_licence" || v === "license" || v === "licence" || v === "dl")
    return "driving_license";
  if (v === "resume" || v === "cv") return "resume";
  return null;
}

function getSelectedDocId() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = normalizeDocId(params.get("doc"));
  if (fromQuery) {
    try {
      localStorage.setItem(DOC_STORAGE_KEY, fromQuery);
    } catch (_) {
      // ignore storage errors
    }
    return fromQuery;
  }

  try {
    return normalizeDocId(localStorage.getItem(DOC_STORAGE_KEY));
  } catch (_) {
    return null;
  }
}

function docLabel(docId) {
  if (docId === "pan") return "PAN";
  if (docId === "aadhaar") return "Aadhaar";
  if (docId === "voter_id") return "Voter ID";
  if (docId === "driving_license") return "Driving License";
  if (docId === "resume") return "Resume";
  return "";
}

function setupSelectedDocUi(selectedDocId) {
  const badge = $("selectedDocBadge");
  if (!badge) return;

  if (!selectedDocId) {
    badge.hidden = true;
    badge.textContent = "";
    return;
  }

  badge.hidden = false;
  badge.textContent = `Selected: ${docLabel(selectedDocId)}`;
}

function setupPortalMismatchModal() {
  const modal = $("portalMismatchModal");
  if (!modal) return null;

  const desc = $("portalMismatchDesc");
  const closeBtn = $("portalMismatchClose");
  const switchLink = $("portalMismatchSwitch");

  let previousOverflow = null;

  function close() {
    modal.hidden = true;
    if (previousOverflow !== null) document.body.style.overflow = previousOverflow;
    previousOverflow = null;
  }

  function open(selectedDocId, detectedDocId) {
    const selectedLabel = docLabel(selectedDocId);
    const detectedLabel = docLabel(detectedDocId);
    if (!selectedLabel || !detectedLabel) return;

    if (desc) {
      desc.textContent = `You selected ${selectedLabel}, but the uploaded image looks like ${detectedLabel}. Please upload the correct document for this portal.`;
    }

    if (switchLink) {
      switchLink.href = `/app?doc=${encodeURIComponent(detectedDocId)}`;
      switchLink.textContent = `Switch to ${detectedLabel}`;
    }

    previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    modal.hidden = false;
    if (closeBtn) closeBtn.focus();
  }

  if (closeBtn) closeBtn.addEventListener("click", close);

  modal.addEventListener("click", (e) => {
    const target = e.target;
    if (target && target.closest && target.closest("[data-modal-close='true']")) close();
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) close();
  });

  return { open, close };
}

function setError(el, message) {
  if (!message) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = message;
}

function pretty(obj) {
  if (typeof obj === "string") return obj;
  return JSON.stringify(obj, null, 2);
}

async function safeJson(res) {
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return await res.json();
  return { detail: await res.text() };
}

function setupChat() {
  const form = $("chatForm");
  const message = $("chatMessage");
  const model = $("chatModel");
  const useDocument = $("chatUseDocument");
  const docStatus = $("chatDocStatus");
  const systemPrompt = $("chatSystem");
  const template = $("chatTemplate");
  const sendBtn = $("chatSend");
  const clearBtn = $("chatClear");

  const outPrompt = $("chatPromptOut");
  const outResponse = $("chatResponseOut");
  const outError = $("chatError");

  function updateDocStatus(doc) {
    if (!doc || typeof doc !== "object") {
      docStatus.textContent = "No document loaded";
      useDocument.checked = false;
      return;
    }

    const classId = typeof doc.class_id === "string" ? doc.class_id : "";
    docStatus.textContent = classId ? `Loaded: ${classId}` : "Document loaded";
    useDocument.checked = true;
  }

  window.addEventListener("document-json", (e) => {
    updateDocStatus(e.detail);
  });

  updateDocStatus(getLastDocumentJson());

  clearBtn.addEventListener("click", () => {
    message.value = "";
    model.value = "";
    systemPrompt.value = "";
    template.value = "";
    outPrompt.textContent = "";
    outResponse.textContent = "";
    setError(outError, "");
    message.focus();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setError(outError, "");
    outPrompt.textContent = "";
    outResponse.textContent = "";

    const msg = (message.value || "").trim();
    if (!msg) {
      setError(outError, "Message is required.");
      return;
    }

    let url = "/chat";
    const payload = { message: msg };

    if (useDocument.checked) {
      const doc = getLastDocumentJson();
      if (!doc) {
        setError(outError, "Parse an image first so the document JSON is available.");
        return;
      }
      url = "/document-chat";
      payload.document_json = doc;
      if ((model.value || "").trim()) payload.model = model.value.trim();
    } else {
      if ((model.value || "").trim()) payload.model = model.value.trim();
      if ((systemPrompt.value || "").trim()) payload.system_prompt = systemPrompt.value.trim();
      if ((template.value || "").trim()) payload.prompt_template = template.value;
    }

    sendBtn.disabled = true;
    sendBtn.textContent = "Sending...";
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await safeJson(res);
      if (!res.ok) {
        setError(outError, data?.detail || `Request failed (${res.status})`);
        return;
      }

      outPrompt.textContent = data.final_prompt || "";
      outResponse.textContent = data.model_response || "";
    } catch (err) {
      setError(outError, err?.message || String(err));
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
    }
  });
}

function setupParse(selectedDocId, portalMismatchModal) {
  const form = $("parseForm");
  const fileInput = $("imageFile");
  const previewImg = $("imagePreview");
  const previewHint = $("previewHint");
  const loading = $("parseLoading");
  const sendBtn = $("parseSend");
  const clearBtn = $("parseClear");
  const fileLabel = $("imageFileLabel");

  const outJson = $("parseJsonOut");
  const outError = $("parseError");

  let previewUrl = null;
  const label = docLabel(selectedDocId);

  if (label) {
    if (fileLabel) fileLabel.textContent = `${label} file (image or PDF)`;
    if (previewHint) previewHint.textContent = `Choose a ${label} image to preview it here (PDF preview is not shown).`;
    document.title = `${label} Upload Â· Document Parser + Ollama Chat`;
  }

  function setLoading(isLoading) {
    if (!loading) return;
    loading.hidden = !isLoading;
  }

  function clearPreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    previewImg.src = "";
    previewImg.style.display = "none";
    previewHint.hidden = false;
  }

  fileInput.addEventListener("change", () => {
    setLoading(false);
    setError(outError, "");
    outJson.textContent = "";
    setLastDocumentJson(null);
    clearPreview();

    const file = fileInput.files && fileInput.files[0];
    if (!file) return;

    const filename = String(file.name || "").toLowerCase();
    const fileType = String(file.type || "").toLowerCase();
    const isPdf = fileType === "application/pdf" || filename.endsWith(".pdf");

    if (isPdf) {
      previewHint.textContent = `PDF selected: ${file.name || "document.pdf"}. Preview is not shown. Click Parse to continue.`;
      previewHint.hidden = false;
      previewImg.src = "";
      previewImg.style.display = "none";
      return;
    } else {
      previewUrl = URL.createObjectURL(file);
      previewImg.src = previewUrl;
      previewImg.style.display = "block";
      previewHint.hidden = true;
    }
  });

  clearBtn.addEventListener("click", () => {
    setLoading(false);
    fileInput.value = "";
    outJson.textContent = "";
    setError(outError, "");
    setLastDocumentJson(null);
    clearPreview();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setError(outError, "");
    outJson.textContent = "";
    setLastDocumentJson(null);

    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      setError(outError, "Please select an image.");
      return;
    }

    const fd = new FormData();
    fd.append("file", file, file.name);
    if (selectedDocId) fd.append("doc_type", selectedDocId);

    setLoading(true);
    sendBtn.disabled = true;
    sendBtn.textContent = "Parsing...";
    try {
      const res = await fetch("/parse", { method: "POST", body: fd });
      const data = await safeJson(res);
      if (!res.ok) {
        setError(outError, data?.detail || `Request failed (${res.status})`);
        return;
      }

      const detectedDocId = normalizeDocId(data?.detected_doc_type || data?.class_id);
      if (data?.mismatch) {
        setLastDocumentJson(null);
        outJson.textContent = pretty(data);
        if (selectedDocId && detectedDocId && detectedDocId !== selectedDocId) {
          portalMismatchModal?.open(selectedDocId, detectedDocId);
        }
        return;
      }

      setLastDocumentJson(data);
      outJson.textContent = pretty(data);
      if (selectedDocId && detectedDocId && detectedDocId !== selectedDocId) {
        portalMismatchModal?.open(selectedDocId, detectedDocId);
      }
    } catch (err) {
      setError(outError, err?.message || String(err));
    } finally {
      setLoading(false);
      sendBtn.disabled = false;
      sendBtn.textContent = "Parse";
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  const selectedDocId = getSelectedDocId();
  setupSelectedDocUi(selectedDocId);
  const portalMismatchModal = setupPortalMismatchModal();
  setupChat();
  setupParse(selectedDocId, portalMismatchModal);
});
