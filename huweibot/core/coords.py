from __future__ import annotations

def _clamp(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _validate_screen(screen_w: int, screen_h: int) -> None:
    if int(screen_w) <= 1 or int(screen_h) <= 1:
        raise ValueError(f"invalid screen size: screen_w={screen_w}, screen_h={screen_h}, both must be > 1")


def _validate_grid_size(grid_size: int) -> None:
    if int(grid_size) <= 0:
        raise ValueError(f"invalid grid_size={grid_size}, must be > 0")


def _validate_grid_dims(grid_w: int, grid_h: int) -> None:
    if int(grid_w) <= 1 or int(grid_h) <= 1:
        raise ValueError(f"invalid grid dims: grid_w={grid_w}, grid_h={grid_h}, both must be > 1")


def _validate_bbox(screen_bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = screen_bbox
    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix2 = int(round(x2))
    iy2 = int(round(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError(f"invalid screen_bbox={screen_bbox}, expected x2>x1 and y2>y1")
    return ix1, iy1, ix2, iy2


def screen_px_to_grid(
    x: int,
    y: int,
    screen_w: int,
    screen_h: int,
    grid_size: int = 1000,
) -> tuple[int, int]:
    _validate_screen(screen_w, screen_h)
    _validate_grid_size(grid_size)

    max_x = int(screen_w) - 1
    max_y = int(screen_h) - 1
    px_x = _clamp(int(round(x)), 0, max_x)
    px_y = _clamp(int(round(y)), 0, max_y)

    gx = round((px_x / max_x) * int(grid_size))
    gy = round((px_y / max_y) * int(grid_size))
    gx = _clamp(int(gx), 0, int(grid_size))
    gy = _clamp(int(gy), 0, int(grid_size))
    return gx, gy


def grid_to_screen_px(
    gx: int,
    gy: int,
    screen_w: int,
    screen_h: int,
    grid_size: int = 1000,
) -> tuple[int, int]:
    _validate_screen(screen_w, screen_h)
    _validate_grid_size(grid_size)

    max_x = int(screen_w) - 1
    max_y = int(screen_h) - 1

    clamped_gx = _clamp(int(round(gx)), 0, int(grid_size))
    clamped_gy = _clamp(int(round(gy)), 0, int(grid_size))

    x = round((clamped_gx / int(grid_size)) * max_x)
    y = round((clamped_gy / int(grid_size)) * max_y)
    x = _clamp(int(x), 0, max_x)
    y = _clamp(int(y), 0, max_y)
    return x, y


def grid_to_screen_px_phone(
    gx: int,
    gy: int,
    screen_bbox: tuple[float, float, float, float],
    grid_w: int = 200,
    grid_h: int = 100,
) -> tuple[int, int]:
    _validate_grid_dims(grid_w, grid_h)
    x1, y1, x2, y2 = _validate_bbox(screen_bbox)
    max_gx = int(grid_w) - 1
    max_gy = int(grid_h) - 1
    cgx = _clamp(int(round(gx)), 0, max_gx)
    cgy = _clamp(int(round(gy)), 0, max_gy)

    x = round((cgx / max_gx) * (x2 - x1)) + x1
    y = round((cgy / max_gy) * (y2 - y1)) + y1
    x = _clamp(int(x), x1, x2)
    y = _clamp(int(y), y1, y2)
    return x, y


def screen_px_to_grid_phone(
    x: int,
    y: int,
    screen_bbox: tuple[float, float, float, float],
    grid_w: int = 200,
    grid_h: int = 100,
) -> tuple[int, int]:
    _validate_grid_dims(grid_w, grid_h)
    x1, y1, x2, y2 = _validate_bbox(screen_bbox)
    max_gx = int(grid_w) - 1
    max_gy = int(grid_h) - 1
    cx = _clamp(int(round(x)), x1, x2)
    cy = _clamp(int(round(y)), y1, y2)

    gx = round(((cx - x1) / max(1, (x2 - x1))) * max_gx)
    gy = round(((cy - y1) / max(1, (y2 - y1))) * max_gy)
    gx = _clamp(int(gx), 0, max_gx)
    gy = _clamp(int(gy), 0, max_gy)
    return gx, gy


def phone_grid_to_screen_px(
    gx: int,
    gy: int,
    screen_bbox: tuple[float, float, float, float],
    grid_w: int = 200,
    grid_h: int = 100,
) -> tuple[int, int]:
    return grid_to_screen_px_phone(gx, gy, screen_bbox, grid_w=grid_w, grid_h=grid_h)


def screen_px_to_phone_grid(
    x: int,
    y: int,
    screen_bbox: tuple[float, float, float, float],
    grid_w: int = 200,
    grid_h: int = 100,
) -> tuple[int, int]:
    return screen_px_to_grid_phone(x, y, screen_bbox, grid_w=grid_w, grid_h=grid_h)


def _self_test() -> None:
    screen_w, screen_h, grid_size = 1920, 1080, 1000
    samples = [
        (0, 0),
        (screen_w - 1, 0),
        (0, screen_h - 1),
        (screen_w - 1, screen_h - 1),
        (123, 456),
        (960, 540),
        (5000, -100),
    ]

    for x, y in samples:
        gx, gy = screen_px_to_grid(x, y, screen_w, screen_h, grid_size)
        rx, ry = grid_to_screen_px(gx, gy, screen_w, screen_h, grid_size)
        if abs(rx - _clamp(x, 0, screen_w - 1)) > 1:
            raise AssertionError(f"round-trip x error too large: x={x}, rx={rx}")
        if abs(ry - _clamp(y, 0, screen_h - 1)) > 1:
            raise AssertionError(f"round-trip y error too large: y={y}, ry={ry}")

    phone_bbox = (100.0, 200.0, 900.0, 1800.0)
    for px, py in [(100, 200), (900, 1800), (512, 999), (9999, -1)]:
        gx, gy = screen_px_to_grid_phone(px, py, phone_bbox, 200, 100)
        rx, ry = grid_to_screen_px_phone(gx, gy, phone_bbox, 200, 100)
        if not (100 <= rx <= 900 and 200 <= ry <= 1800):
            raise AssertionError("phone round-trip out of bbox")

    print("OK")


if __name__ == "__main__":
    _self_test()
