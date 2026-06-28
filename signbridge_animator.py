"""
SignBridge Text-to-Sign Animator
---------------------------------
- Type words directly on screen (no terminal)
- Fake head from shoulder midpoint
- Continuous sequence with interpolation between words
- Cache for instant startup after first run
Controls: Type words → ENTER to animate → ESC to stop → BACKSPACE to edit
"""

import os
import numpy as np
import pygame

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR   = r"D:\Extracted"
CACHE_FILE = r"vocab_cache.npy"
FPS        = 20
WINDOW_W, WINDOW_H = 900, 700
TRANSITION_FRAMES  = 8

BG_COLOR     = (15, 15, 25)
POSE_COLOR   = (0, 220, 180)
LHAND_COLOR  = (80, 180, 255)
RHAND_COLOR  = (255, 120, 80)
HEAD_COLOR   = (0, 220, 180)
INPUT_BG     = (25, 25, 40)
INPUT_BORDER = (60, 60, 100)
INPUT_ACTIVE = (0, 180, 140)
HINT_COLOR   = (80, 80, 110)
ERROR_COLOR  = (220, 80, 80)
JOINT_RADIUS = 5
BONE_WIDTH   = 2
HEAD_RADIUS  = 28

POSE_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(3,5)
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ─── VOCAB ────────────────────────────────────────────────────────────────────

def pick_representative(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    if not files:
        return None
    seqs = []
    for f in files:
        try:
            s = np.load(os.path.join(folder_path, f))
            if s.ndim == 2 and s.shape[1] == 144:
                seqs.append((len(s), f))
        except Exception:
            continue
    if not seqs:
        return None
    seqs.sort(key=lambda x: x[0])
    return np.load(os.path.join(folder_path, seqs[len(seqs)//2][1]))


def load_vocab():
    if os.path.exists(CACHE_FILE):
        print("Cache found — loading...")
        v = np.load(CACHE_FILE, allow_pickle=True).item()
        print(f"  ✓ {len(v)} words ready")
        return v
    vocab = {}
    classes = os.listdir(DATA_DIR)
    print(f"First run — building cache ({len(classes)} classes)...")
    for cls in classes:
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(folder):
            continue
        seq = pick_representative(folder)
        if seq is not None:
            vocab[cls.lower()] = seq
            print(f"  ✓ {cls}")
    np.save(CACHE_FILE, vocab)
    print("  → Cache saved.")
    return vocab

# ─── COORDS ───────────────────────────────────────────────────────────────────

def extract_frame(f144):
    return f144[0:18].reshape(6,3), f144[18:81].reshape(21,3), f144[81:144].reshape(21,3)

def compute_bounds(sequences):
    all_pts = np.vstack([s.reshape(-1,3) for s in sequences])
    valid   = all_pts[np.any(all_pts != 0, axis=1)]
    if len(valid) == 0:
        return None
    return valid.min(axis=0), valid.max(axis=0)

def to_screen(pts, bounds, margin=80):
    mn, mx = bounds
    rng = mx - mn
    rng[rng == 0] = 1
    norm = (pts - mn) / rng
    xs = norm[:,0] * (WINDOW_W - 2*margin) + margin
    ys = norm[:,1] * (WINDOW_H - 2*margin - 120) + margin + 60  # leave room for UI
    return np.stack([xs, ys], axis=1).astype(int)

# ─── DRAW ─────────────────────────────────────────────────────────────────────

def draw_skeleton(surface, pose_px, lhand_px, rhand_px):
    mid_x = int((pose_px[0][0] + pose_px[1][0]) / 2)
    mid_y = int((pose_px[0][1] + pose_px[1][1]) / 2)
    head_center = (mid_x, mid_y - HEAD_RADIUS - 10)
    pygame.draw.line(surface, POSE_COLOR, (mid_x, mid_y), head_center, BONE_WIDTH+1)
    pygame.draw.circle(surface, HEAD_COLOR, head_center, HEAD_RADIUS, 2)

    for a,b in POSE_CONNECTIONS:
        pygame.draw.line(surface, POSE_COLOR, tuple(pose_px[a]), tuple(pose_px[b]), BONE_WIDTH)
    for pt in pose_px:
        pygame.draw.circle(surface, POSE_COLOR, tuple(pt), JOINT_RADIUS)

    for a,b in HAND_CONNECTIONS:
        pygame.draw.line(surface, LHAND_COLOR, tuple(lhand_px[a]), tuple(lhand_px[b]), BONE_WIDTH)
    for pt in lhand_px:
        pygame.draw.circle(surface, LHAND_COLOR, tuple(pt), JOINT_RADIUS-1)

    for a,b in HAND_CONNECTIONS:
        pygame.draw.line(surface, RHAND_COLOR, tuple(rhand_px[a]), tuple(rhand_px[b]), BONE_WIDTH)
    for pt in rhand_px:
        pygame.draw.circle(surface, RHAND_COLOR, tuple(pt), JOINT_RADIUS-1)


def draw_word_strip(surface, font_sm, words, current_idx):
    """Top bar showing word sequence, current highlighted."""
    y = 12
    x = 20
    for i, w in enumerate(words):
        color = (255, 255, 255) if i == current_idx else (60, 60, 90)
        surf  = font_sm.render(w.upper(), True, color)
        surface.blit(surf, (x, y))
        x += surf.get_width() + 20


def draw_progress(surface, global_frame, total_frames):
    bar_w  = WINDOW_W - 40
    bar_h  = 5
    bar_x  = 20
    bar_y  = WINDOW_H - 70
    filled = int(bar_w * global_frame / max(total_frames-1, 1))
    pygame.draw.rect(surface, (35,35,55), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
    pygame.draw.rect(surface, POSE_COLOR, (bar_x, bar_y, filled, bar_h), border_radius=3)


def draw_input_box(surface, font, font_sm, text, error_msg, vocab_keys):
    box_h  = 50
    box_y  = WINDOW_H - box_h - 8
    box_x  = 20
    box_w  = WINDOW_W - 40

    pygame.draw.rect(surface, INPUT_BG,     (box_x, box_y, box_w, box_h), border_radius=8)
    pygame.draw.rect(surface, INPUT_ACTIVE, (box_x, box_y, box_w, box_h), 2, border_radius=8)

    # Prompt
    prompt = font_sm.render("▶ ", True, INPUT_ACTIVE)
    surface.blit(prompt, (box_x+10, box_y+15))

    # Typed text — colour each word green/red based on vocab match
    words = text.split(" ")
    cx = box_x + 10 + prompt.get_width()
    for i, w in enumerate(words):
        if w == "":
            continue
        color = POSE_COLOR if w.lower() in vocab_keys else ERROR_COLOR
        ws = font.render(w, True, color)
        surface.blit(ws, (cx, box_y+13))
        cx += ws.get_width()
        # space
        sp = font.render(" ", True, (255,255,255))
        surface.blit(sp, (cx, box_y+13))
        cx += sp.get_width()

    # Cursor blink
    if (pygame.time.get_ticks() // 500) % 2 == 0:
        pygame.draw.rect(surface, (200,200,200), (cx, box_y+14, 2, 22))

    # Hint or error below box
    if error_msg:
        err = font_sm.render(error_msg, True, ERROR_COLOR)
        surface.blit(err, (box_x+10, box_y - 22))
    else:
        hint = font_sm.render("ENTER = sign   BACKSPACE = delete   ESC = clear", True, HINT_COLOR)
        surface.blit(hint, (box_x+10, box_y - 22))

    # Legend
    surface.blit(font_sm.render("■ Pose",   True, POSE_COLOR),  (WINDOW_W-130, WINDOW_H-65))
    surface.blit(font_sm.render("■ L.Hand", True, LHAND_COLOR), (WINDOW_W-130, WINDOW_H-50))
    surface.blit(font_sm.render("■ R.Hand", True, RHAND_COLOR), (WINDOW_W-130, WINDOW_H-35))

# ─── SEQUENCE ─────────────────────────────────────────────────────────────────

def build_continuous(words, vocab):
    result = []
    seqs   = [vocab[w] for w in words]
    for i, seq in enumerate(seqs):
        for frame in seq:
            result.append((frame, i))
        if i < len(seqs)-1:
            last  = seq[-1]
            first = seqs[i+1][0]
            for t in range(1, TRANSITION_FRAMES+1):
                alpha  = t / (TRANSITION_FRAMES+1)
                interp = (1-alpha)*last + alpha*first
                result.append((interp, i))
    return result

# ─── STATES ───────────────────────────────────────────────────────────────────

STATE_INPUT   = "input"
STATE_ANIMATE = "animate"

def main():
    print("Loading vocab...")
    vocab = load_vocab()
    vocab_keys = set(vocab.keys())
    print(f"Ready. Known words: {', '.join(sorted(vocab_keys))}")

    pygame.init()
    screen   = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("SignBridge Animator")
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont("consolas", 22)
    font_sm  = pygame.font.SysFont("consolas", 15)
    font_big = pygame.font.SysFont("consolas", 28, bold=True)

    state     = STATE_INPUT
    typed     = ""
    error_msg = ""
    sequence  = []
    words     = []
    bounds    = None
    frame_idx = 0
    paused    = False

    while True:
        # ── EVENTS ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if state == STATE_INPUT:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        typed     = ""
                        error_msg = ""

                    elif event.key == pygame.K_BACKSPACE:
                        typed     = typed[:-1]
                        error_msg = ""

                    elif event.key == pygame.K_RETURN:
                        # Parse and validate
                        raw     = typed.strip().split()
                        matched = [w.lower() for w in raw if w.lower() in vocab_keys]
                        skipped = [w for w in raw if w.lower() not in vocab_keys]

                        if not matched:
                            error_msg = f"No known words. Unknown: {skipped}" if skipped else "Type some words first."
                        else:
                            if skipped:
                                error_msg = ""  # will show in strip
                            seqs     = [vocab[w] for w in matched]
                            bounds   = compute_bounds(seqs)
                            sequence = build_continuous(matched, vocab)
                            words    = matched
                            frame_idx = 0
                            paused    = False
                            state     = STATE_ANIMATE
                            typed     = ""
                            error_msg = ""

                    elif event.key == pygame.K_SPACE:
                        typed += " "

                    else:
                        if event.unicode.isprintable() and event.unicode != " ":
                            typed += event.unicode
                            error_msg = ""

                elif event.type == pygame.TEXTINPUT:
                    # TEXTINPUT fires for printable chars — use KEYDOWN above instead
                    pass

            elif state == STATE_ANIMATE:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        state     = STATE_INPUT
                        typed     = ""
                        error_msg = ""
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

        # ── DRAW ──
        screen.fill(BG_COLOR)

        if state == STATE_INPUT:
            # Title
            title = font_big.render("SignBridge", True, POSE_COLOR)
            screen.blit(title, (WINDOW_W//2 - title.get_width()//2, WINDOW_H//2 - 80))

            sub = font_sm.render("Type words below and press ENTER to animate", True, HINT_COLOR)
            screen.blit(sub, (WINDOW_W//2 - sub.get_width()//2, WINDOW_H//2 - 40))

            # Available words hint
            avail = "  ".join(sorted(vocab_keys))
            av_surf = font_sm.render(f"Available: {avail}", True, (45,45,70))
            screen.blit(av_surf, (20, WINDOW_H//2 + 10))

            draw_input_box(screen, font, font_sm, typed, error_msg, vocab_keys)

        elif state == STATE_ANIMATE:
            if frame_idx < len(sequence):
                frame_144, word_idx = sequence[frame_idx]
                pose_3d, lhand_3d, rhand_3d = extract_frame(frame_144)
                all_pts = np.vstack([pose_3d, lhand_3d, rhand_3d])
                all_px  = to_screen(all_pts, bounds)

                draw_skeleton(screen, all_px[0:6], all_px[6:27], all_px[27:48])
                draw_word_strip(screen, font, words, word_idx)
                draw_progress(screen, frame_idx, len(sequence))

                # Controls hint
                ctrl = font_sm.render("SPACE = pause   ESC = back to input", True, HINT_COLOR)
                screen.blit(ctrl, (20, WINDOW_H - 50))

                if not paused:
                    frame_idx += 1
            else:
                # Done — go back to input
                state     = STATE_INPUT
                typed     = ""
                error_msg = ""

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()