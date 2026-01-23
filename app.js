const status = document.getElementById("status");
const content = document.getElementById("content");
const timerEl = document.getElementById("timer");

const WS_URL = `ws://${window.location.hostname}:9001/ws`;
const ws = new WebSocket(WS_URL);

let expiry = 0;
let filename = "";

ws.onopen = () => {
    status.innerText = "CONNECTED";
    status.style.color = "lime";
};

ws.onerror = () => {
    status.innerText = "ERROR";
    status.style.color = "orange";
};

ws.onclose = () => {
    status.innerText = "DISCONNECTED";
    status.style.color = "red";
};

ws.onmessage = async (event) => {
    if (typeof event.data === "string") {
        filename = event.data;
        return;
    }

    const blob = new Blob([event.data]);
    const url = URL.createObjectURL(blob);
    expiry = Date.now() + 20000;

    renderFile(url, filename);
    startTimer();
};

function renderFile(url, name) {
    content.innerHTML = "";

    if (name.match(/\.(jpg|png|gif)$/i)) {
        content.innerHTML = `<img src="${url}">`;
    } else if (name.match(/\.(mp4|webm)$/i)) {
        content.innerHTML = `<video src="${url}" controls autoplay></video>`;
    } else if (name.match(/\.(mp3|wav|ogg)$/i)) {
        content.innerHTML = `<audio src="${url}" controls autoplay></audio>`;
    } else {
        content.innerHTML = `<a href="${url}" download>Download ${name}</a>`;
    }
}

function startTimer() {
    const interval = setInterval(() => {
        const remaining = Math.max(0, Math.floor((expiry - Date.now()) / 1000));
        timerEl.innerText = remaining > 0 ? `Expires in ${remaining}s` : "";

        if (remaining <= 0) {
            content.innerHTML = "";
            clearInterval(interval);
        }
    }, 1000);
}
