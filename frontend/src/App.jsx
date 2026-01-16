import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const STREAM_URL = import.meta.env.VITE_BROWSER_STREAM_URL || "";

const SCRUB_STATUS = {
  idle: "Blank",
  scrubbing: "Scrubbing",
  complete: "Return to Blank",
};

const SCRUB_CODES = [
  "wipe.cookies()",
  "dissolve.identity()",
  "scrub.session()",
  "purge.traces()",
  "clean.room()",
  "erase.cache()",
  "mask.profile()",
  "sanitize.logs()",
];

function BubbleCanvas({ isScrubbing }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
    });
    renderer.setPixelRatio(window.devicePixelRatio || 1);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 4);

    const geometry = new THREE.SphereGeometry(1.2, 64, 64);
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0xffffff),
      roughness: 0.08,
      metalness: 0.05,
      transmission: 1,
      thickness: 0.4,
      ior: 1.35,
      opacity: 0.85,
      transparent: true,
      iridescence: 0.8,
      iridescenceIOR: 1.3,
      iridescenceThicknessRange: [100, 400],
      clearcoat: 1,
      clearcoatRoughness: 0.2,
    });

    const bubble = new THREE.Mesh(geometry, material);
    scene.add(bubble);

    const ambient = new THREE.AmbientLight(0x8dffcc, 0.55);
    scene.add(ambient);

    const keyLight = new THREE.PointLight(0xffffff, 1.1, 10);
    keyLight.position.set(3, 2, 4);
    scene.add(keyLight);

    const fillLight = new THREE.PointLight(0x9f7bff, 0.9, 8);
    fillLight.position.set(-3, -2, 3);
    scene.add(fillLight);

    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      if (width === 0 || height === 0) {
        return;
      }
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };

    resize();
    window.addEventListener("resize", resize);

    let frameId;
    const animate = (time) => {
      const wobble = 1 + Math.sin(time * 0.0015) * 0.03 + (isScrubbing ? 0.06 : 0);
      bubble.rotation.y = time * 0.0004;
      bubble.rotation.x = time * 0.00025;
      bubble.scale.setScalar(wobble);
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };

    frameId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", resize);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, [isScrubbing]);

  return <canvas ref={canvasRef} className="h-full w-full" />;
}

function App() {
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [isPopped, setIsPopped] = useState(false);
  const [status, setStatus] = useState(SCRUB_STATUS.idle);
  const [pending, setPending] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "system", text: "Solvent online. Return to Blank." },
    { role: "agent", text: "Awaiting instructions." },
  ]);

  const codeLines = useMemo(() => SCRUB_CODES, []);

  const appendMessage = (role, text) => {
    setMessages((prev) => [...prev, { role, text }]);
  };

  const triggerPop = () => {
    setIsPopped(true);
    window.setTimeout(() => setIsPopped(false), 700);
  };

  const runScrub = async () => {
    if (pending) {
      return;
    }
    setPending(true);
    setIsScrubbing(true);
    setStatus(SCRUB_STATUS.scrubbing);
    appendMessage("agent", "Bubble blower engaged. Scrub in progress.");

    try {
      const response = await fetch(`${API_BASE}/scrub/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        const errorText = await response.text();
        appendMessage("system", `Scrub halted: ${errorText}`);
      } else {
        appendMessage("agent", "Scrub complete. Returning to blank.");
      }
    } catch (error) {
      appendMessage("system", `API offline. ${String(error)}`);
    } finally {
      setIsScrubbing(false);
      setStatus(SCRUB_STATUS.complete);
      triggerPop();
      window.setTimeout(() => setStatus(SCRUB_STATUS.idle), 1200);
      setPending(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) {
      return;
    }
    appendMessage("user", trimmed);
    appendMessage("agent", "Queued. Awaiting execution.");
    setInput("");
  };

  return (
    <div className="min-h-screen bg-solvent-black text-solvent-neon">
      <header className="flex items-center justify-between px-8 py-6">
        <div className="flex items-center gap-4">
          <div className="text-4xl font-semibold chrome-logo">S</div>
          <div>
            <div className="text-lg tracking-[0.35em] uppercase">Solvent</div>
            <div className="text-xs uppercase tracking-[0.5em] text-solvent-steel">
              Return to Blank
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={runScrub}
          disabled={pending}
          className="rounded-full border border-solvent-neon/40 px-6 py-2 text-xs uppercase tracking-[0.4em] shadow-glow transition hover:border-solvent-neon hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {pending ? "Scrubbing" : "Start Scrub"}
        </button>
      </header>

      <main className="flex min-h-[calc(100vh-96px)] flex-col gap-6 px-8 pb-8">
        <section className="flex-[7]">
          <div className="bubble-shell relative h-full min-h-[60vh] rounded-[32px] p-6">
            <div className="absolute inset-0 rounded-[32px] bg-gradient-to-br from-white/5 via-transparent to-indigo-500/10" />
            <div className="relative z-10 grid h-full gap-6 lg:grid-cols-[1.4fr_0.8fr]">
              <div className="flex h-full flex-col">
                <div className="flex items-center justify-between text-xs uppercase tracking-[0.4em] text-solvent-steel">
                  <span>Live Browser View</span>
                  <span className="text-solvent-neon">{status}</span>
                </div>
                <div className="mt-4 flex-1 rounded-3xl border border-solvent-haze/80 bg-black/60 p-4">
                  {STREAM_URL ? (
                    <iframe
                      title="Live browser stream"
                      src={STREAM_URL}
                      className="h-full w-full rounded-2xl border border-solvent-haze/60 bg-black/80"
                    />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center rounded-2xl border border-solvent-haze/60 bg-black/80 text-sm uppercase tracking-[0.35em] text-solvent-steel">
                      Connect Playwright stream
                    </div>
                  )}
                </div>
              </div>

              <div className="relative flex h-full items-center justify-center">
                <div
                  className={`relative aspect-square w-full max-w-[320px] rounded-full ${
                    isScrubbing ? "animate-inflate" : "animate-floaty"
                  } ${isPopped ? "animate-pop" : ""}`}
                >
                  <BubbleCanvas isScrubbing={isScrubbing} />
                  {isScrubbing && (
                    <div className="code-stream pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 text-[10px] uppercase tracking-[0.35em]">
                      {codeLines.map((line) => (
                        <span key={line} className="animate-codepulse">
                          {line}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="pointer-events-none absolute -bottom-8 left-1/2 h-10 w-28 -translate-x-1/2 rounded-full bg-gradient-to-r from-transparent via-solvent-neon/30 to-transparent blur-lg" />
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="flex-[3]">
          <div className="blur-mask flex h-full flex-col rounded-[28px] border border-solvent-haze/80 bg-black/55 p-6">
            <div className="text-xs uppercase tracking-[0.4em] text-solvent-steel">
              Agent Console
            </div>
            <div className="code-stream mt-4 flex-1 space-y-3 overflow-y-auto text-sm">
              {messages.map((message, index) => (
                <div key={`${message.role}-${index}`}>
                  <span className="mr-2 text-solvent-steel">{message.role}:</span>
                  <span>{message.text}</span>
                </div>
              ))}
            </div>
            <form
              onSubmit={handleSubmit}
              className="mt-4 flex items-center gap-3 rounded-full border border-solvent-haze/70 bg-black/70 px-4 py-2"
            >
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Issue a scrub command..."
                className="flex-1 bg-transparent text-sm uppercase tracking-[0.25em] text-solvent-neon outline-none placeholder:text-solvent-steel/80"
              />
              <button
                type="submit"
                className="rounded-full border border-solvent-neon/40 px-4 py-1 text-[10px] uppercase tracking-[0.35em] transition hover:border-solvent-neon hover:text-white"
              >
                Send
              </button>
            </form>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
