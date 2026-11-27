import { useEffect, useRef } from "react";

const PARTICLE_COLORS = [
  "#00D4FF", "#00D4FF", "#00D4FF",
  "#A78BFA", "#A78BFA", "#A78BFA",
  "#F59E0B", "#F59E0B",
  "#FFFFFF", "#FFFFFF",
];

export function BackgroundScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();

    const particles = Array.from({ length: 72 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 1.5 + 0.8,
      color: PARTICLE_COLORS[Math.floor(Math.random() * PARTICLE_COLORS.length)],
      opacity: Math.random() * 0.3 + 0.15,
      vx: (Math.random() - 0.5) * 0.12,
      vy: (Math.random() - 0.5) * 0.12,
      glow: Math.random() > 0.55,
    }));

    let animId: number;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.save();
        ctx.globalAlpha = p.opacity;

        if (p.glow) {
          const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 5);
          const hex = p.color;
          g.addColorStop(0, hex + "55");
          g.addColorStop(1, "transparent");
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 5, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
      }

      animId = requestAnimationFrame(draw);
    };

    draw();
    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden" style={{ zIndex: 0 }}>
      {/* Base deep space gradient — breathes slowly */}
      <div
        className="absolute inset-0"
        style={{
          backgroundColor: "#020510",
          backgroundImage: `
            radial-gradient(ellipse 900px 700px at 15% 12%, rgba(10,22,40,0.95) 0%, transparent 65%),
            radial-gradient(ellipse 700px 700px at 55% 50%, rgba(13,10,46,0.85) 0%, transparent 65%),
            radial-gradient(ellipse 600px 550px at 10% 88%, rgba(5,26,26,0.75) 0%, transparent 65%),
            radial-gradient(ellipse 500px 400px at 88% 88%, rgba(2,5,16,0.90) 0%, transparent 65%)
          `,
          backgroundSize: "200% 200%",
          animation: "drift 22s ease-in-out infinite",
        }}
      />

      {/* Accent light sources */}
      <div
        className="absolute"
        style={{
          top: "-180px", left: "-180px",
          width: "700px", height: "700px",
          background: "radial-gradient(circle, rgba(10,42,110,0.25) 0%, transparent 65%)",
          borderRadius: "50%",
        }}
      />
      <div
        className="absolute"
        style={{
          top: "18%", right: "-160px",
          width: "500px", height: "500px",
          background: "radial-gradient(circle, rgba(26,10,62,0.20) 0%, transparent 65%)",
          borderRadius: "50%",
        }}
      />
      <div
        className="absolute"
        style={{
          bottom: "-120px", left: "28%",
          width: "600px", height: "380px",
          background: "radial-gradient(circle, rgba(10,30,46,0.15) 0%, transparent 65%)",
          borderRadius: "50%",
        }}
      />

      {/* Perspective grid — bottom 40% of screen */}
      <div
        className="absolute bottom-0"
        style={{
          left: "-60%", right: "-60%",
          height: "48%",
          backgroundImage: `
            linear-gradient(rgba(13,32,64,0.65) 1px, transparent 1px),
            linear-gradient(90deg, rgba(13,32,64,0.65) 1px, transparent 1px)
          `,
          backgroundSize: "40px 40px",
          transform: "perspective(500px) rotateX(62deg)",
          transformOrigin: "center bottom",
          maskImage: "linear-gradient(to top, rgba(0,0,0,0.75) 0%, rgba(0,0,0,0.3) 45%, transparent 100%)",
          WebkitMaskImage: "linear-gradient(to top, rgba(0,0,0,0.75) 0%, rgba(0,0,0,0.3) 45%, transparent 100%)",
        }}
      />

      {/* Grid glow dots at intersections */}
      <div
        className="absolute bottom-0"
        style={{
          left: "-60%", right: "-60%",
          height: "48%",
          backgroundImage: "radial-gradient(circle 1.5px at 0px 0px, rgba(0,212,255,0.08) 0%, transparent 100%)",
          backgroundSize: "40px 40px",
          transform: "perspective(500px) rotateX(62deg)",
          transformOrigin: "center bottom",
          maskImage: "linear-gradient(to top, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.2) 45%, transparent 100%)",
          WebkitMaskImage: "linear-gradient(to top, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.2) 45%, transparent 100%)",
        }}
      />

      {/* Floating particle canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ mixBlendMode: "screen", opacity: 0.85 }}
      />
    </div>
  );
}
