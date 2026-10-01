const $ = (id) => document.getElementById(id);

const gestureLabels = {
    THUMBS_UP: "👍 THUMBS_UP",
    THUMBS_DOWN: "👎 THUMBS_DOWN",
    OPEN_PALM: "✋ COMMIT",
    POINTING: "☝ DRAW NUMBER",
    OP_ADD: "PINKY → +",
    OP_SUBTRACT: "PINKY + RING → −",
    OP_MULTIPLY: "PINKY + RING + MIDDLE → ×",
    OP_DIVIDE: "4 FINGERS → ÷",
    OP_EQUALS: "✊ → =",
};

async function post(path) {
    const res = await fetch(path, {method: "POST"});
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

$("startBtn").addEventListener("click", () => post("/api/start"));
$("clearBtn").addEventListener("click", () => post("/api/clear"));
$("stopBtn").addEventListener("click", () => post("/api/stop"));
$("resetFaceBtn").addEventListener("click", async () => {
    if (confirm("چهره ثبت‌شده پاک و چهره جدید ثبت شود؟")) {
        await post("/api/reset-identity");
    }
});

function render(data) {
    $("peopleCount").textContent = data.people_count;
    $("state").textContent = data.state;
    $("gesture").textContent = gestureLabels[data.gesture] || "—";
    $("enrolled").textContent = data.identity_enrolled ? "بله" : "خیر";
    $("verified").textContent = data.identity_verified ? "تأیید شد" : "خیر";
    $("similarity").textContent =
        data.face_similarity == null ? "—" : Number(data.face_similarity).toFixed(3);

    $("buffer").textContent = data.buffer || "—";
    $("result").textContent = data.result ?? "—";

    const tokenBox = $("tokens");
    tokenBox.innerHTML = "";
    if (!data.tokens.length) {
        tokenBox.innerHTML = '<span style="color:#7183a0">هنوز Token ثبت نشده است.</span>';
    } else {
        for (const token of data.tokens) {
            const el = document.createElement("span");
            el.className = "token";
            el.textContent = token;
            tokenBox.appendChild(el);
        }
    }

    const events = $("events");
    events.innerHTML = "";
    for (const event of data.events) {
        const row = document.createElement("div");
        row.className = "event";

        const time = document.createElement("span");
        time.className = "event-time";
        time.textContent = event.time;

        const msg = document.createElement("span");
        msg.className = "event-text";
        msg.textContent = event.message;

        row.append(time, msg);
        events.appendChild(row);
    }

    const badge = $("cameraBadge");
    if (data.camera_ok) {
        badge.textContent = "CAMERA ONLINE";
        badge.className = "badge ok";
    } else {
        badge.textContent = "CAMERA OFFLINE";
        badge.className = "badge danger";
    }
}

async function poll() {
    try {
        const res = await fetch("/api/status", {cache: "no-store"});
        if (res.ok) render(await res.json());
    } catch (_) {}
}

setInterval(poll, 180);
poll();
