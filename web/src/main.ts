import spec from "../../specs/cursor_world.json";
import "./style.css";

type Entity = {
  id: number;
  shape: string;
  color: number[];
  x: number;
  y: number;
  size: number;
};

type Action = { dx: number; dy: number; click: boolean };

class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (1664525 * this.state + 1013904223) >>> 0;
    return this.state / 2 ** 32;
  }

  range(min: number, max: number): number {
    return min + (max - min) * this.next();
  }
}

class CursorWorld {
  entities: Entity[] = [];
  cursorX = spec.cursor.start[0];
  cursorY = spec.cursor.start[1];
  mouseDown = false;
  grabbedId: number | null = null;
  backgroundAlt = false;
  private rng: Rng;

  constructor(seed: number) {
    this.rng = new Rng(seed);
    this.reset();
  }

  reset() {
    this.entities = [];
    const count = Math.floor(this.rng.range(spec.entities.min_count, spec.entities.max_count + 1));
    for (let i = 0; i < count; i += 1) {
      const size = Math.floor(this.rng.range(spec.entities.sizes[0], spec.entities.sizes[1] + 1));
      this.entities.push({
        id: i,
        shape: spec.entities.shapes[i % spec.entities.shapes.length],
        color: spec.entities.colors[i % spec.entities.colors.length],
        x: this.rng.range(size / 2, spec.canvas.width - size / 2),
        y: this.rng.range(size / 2, spec.canvas.height - size / 2),
        size,
      });
    }
  }

  step(action: Action) {
    const dx = clamp(action.dx, -spec.cursor.max_delta, spec.cursor.max_delta);
    const dy = clamp(action.dy, -spec.cursor.max_delta, spec.cursor.max_delta);
    const prevDown = this.mouseDown;
    this.cursorX = clamp(this.cursorX + dx, 0, spec.canvas.width - 1);
    this.cursorY = clamp(this.cursorY + dy, 0, spec.canvas.height - 1);
    this.mouseDown = action.click;

    if (action.click && !prevDown) {
      if (this.cursorInButton()) {
        this.backgroundAlt = !this.backgroundAlt;
      } else {
        this.grabbedId = this.topEntityAtCursor();
      }
    }
    if (!action.click) this.grabbedId = null;

    if (this.grabbedId !== null) {
      const ent = this.entities[this.grabbedId];
      const half = ent.size / 2;
      ent.x = clamp(this.cursorX, half, spec.canvas.width - half);
      ent.y = clamp(this.cursorY, half, spec.canvas.height - half);
    }
  }

  draw(ctx: CanvasRenderingContext2D) {
    const scale = ctx.canvas.width / spec.canvas.width;
    ctx.save();
    ctx.scale(scale, scale);
    ctx.fillStyle = rgb(this.backgroundAlt ? spec.canvas.background_alt : spec.canvas.background);
    ctx.fillRect(0, 0, spec.canvas.width, spec.canvas.height);

    for (const ent of this.entities) {
      ctx.fillStyle = rgb(ent.color);
      const half = ent.size / 2;
      if (ent.shape === "cube") {
        ctx.fillRect(ent.x - half, ent.y - half, ent.size, ent.size);
      } else if (ent.shape === "circle") {
        ctx.beginPath();
        ctx.arc(ent.x, ent.y, half, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.beginPath();
        ctx.moveTo(ent.x, ent.y - half);
        ctx.lineTo(ent.x - half, ent.y + half);
        ctx.lineTo(ent.x + half, ent.y + half);
        ctx.closePath();
        ctx.fill();
      }
    }

    const b = spec.button;
    ctx.strokeStyle = rgb(b.color);
    ctx.strokeRect(b.x, b.y, b.width, b.height);
    ctx.strokeStyle = rgb(spec.cursor.color);
    ctx.beginPath();
    ctx.moveTo(this.cursorX - spec.cursor.size, this.cursorY);
    ctx.lineTo(this.cursorX + spec.cursor.size, this.cursorY);
    ctx.moveTo(this.cursorX, this.cursorY - spec.cursor.size);
    ctx.lineTo(this.cursorX, this.cursorY + spec.cursor.size);
    ctx.stroke();
    ctx.restore();
  }

  private cursorInButton(): boolean {
    const b = spec.button;
    return this.cursorX >= b.x && this.cursorX <= b.x + b.width && this.cursorY >= b.y && this.cursorY <= b.y + b.height;
  }

  private topEntityAtCursor(): number | null {
    for (let i = this.entities.length - 1; i >= 0; i -= 1) {
      const ent = this.entities[i];
      const half = ent.size / 2;
      if (this.cursorX >= ent.x - half && this.cursorX <= ent.x + half && this.cursorY >= ent.y - half && this.cursorY <= ent.y + half) {
        return ent.id;
      }
    }
    return null;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function rgb(v: number[]): string {
  return `rgb(${v[0]}, ${v[1]}, ${v[2]})`;
}

const canvas = document.querySelector<HTMLCanvasElement>("#world");
if (!canvas) throw new Error("missing canvas");
const ctx = canvas.getContext("2d");
if (!ctx) throw new Error("missing 2d context");

const world = new CursorWorld(7);
let lastX = spec.cursor.start[0];
let lastY = spec.cursor.start[1];
let click = false;

canvas.addEventListener("pointerdown", (event) => {
  click = true;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointerup", () => {
  click = false;
});
canvas.addEventListener("pointermove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * spec.canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * spec.canvas.height;
  world.step({ dx: x - lastX, dy: y - lastY, click });
  lastX = x;
  lastY = y;
});

function frame() {
  world.draw(ctx!);
  requestAnimationFrame(frame);
}

frame();

