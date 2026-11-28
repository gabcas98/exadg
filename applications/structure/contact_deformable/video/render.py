"""
Frames of the self-contact test (closing slot) in a paper-like layout:
 (a) deformed specimen, (b) detail of the slot mouth with the contact zone,
 (c) contact length versus prescribed displacement.

usage: python3 -I render.py <runs directory> <frames directory> <maximum push> <number of states>
"""
import glob
import os
import sys

import vtk

runs_dir, frames_dir = sys.argv[1], sys.argv[2]
PUSH_MAX, N = float(sys.argv[3]), int(sys.argv[4])
os.makedirs(frames_dir, exist_ok=True)

Y0, Y1, SLOT_END = 0.49, 0.51, 0.6
WIDTH, HEIGHT = 1920, 1080
FONT_DIR = "/usr/share/texmf/fonts/opentype/public/lm/"
FONT = {"regular": FONT_DIR + "lmroman10-regular.otf",
        "bold": FONT_DIR + "lmroman10-bold.otf",
        "italic": FONT_DIR + "lmroman10-italic.otf"}
BLACK = (0.0, 0.0, 0.0)
GRAY = (0.45, 0.45, 0.45)
LOWER = (0.80, 0.33, 0.0)    # lower face of the slot
UPPER = (0.0, 0.45, 0.70)    # upper face of the slot
CONTACT = (0.80, 0.0, 0.0)
MAX_U = PUSH_MAX

VIRIDIS = [(0.000, (0.267, 0.005, 0.329)), (0.125, (0.283, 0.141, 0.458)),
           (0.250, (0.254, 0.265, 0.530)), (0.375, (0.207, 0.372, 0.553)),
           (0.500, (0.164, 0.471, 0.558)), (0.625, (0.128, 0.567, 0.551)),
           (0.750, (0.135, 0.659, 0.518)), (0.875, (0.267, 0.749, 0.441)),
           (1.000, (0.993, 0.906, 0.144))]


def lookup_table():
    ctf = vtk.vtkColorTransferFunction()
    for t, rgb in VIRIDIS:
        ctf.AddRGBPoint(t * MAX_U, *rgb)
    return ctf


def read(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    lagrange = reader.GetOutput()
    # degree-1 Lagrange quadrilaterals -> VTK_QUAD (corners first), drawn without diagonals
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(lagrange.GetPoints())
    grid.GetPointData().ShallowCopy(lagrange.GetPointData())
    grid.Allocate(lagrange.GetNumberOfCells())
    for i in range(lagrange.GetNumberOfCells()):
        ids = lagrange.GetCell(i).GetPointIds()
        quad = vtk.vtkIdList()
        for k in range(4):
            quad.InsertNextId(ids.GetId(k))
        grid.InsertNextCell(vtk.VTK_QUAD, quad)
    grid.GetPointData().SetActiveVectors("displacement")
    grid.GetPointData().SetActiveScalars("displacement_magnitude")
    return grid


def slot_faces(grid):
    """Deformed positions of the nodes of the lower and upper slot faces, sorted by x."""
    displacement = grid.GetPointData().GetArray("displacement")
    lower, upper = {}, {}
    for i in range(grid.GetNumberOfPoints()):
        X = grid.GetPoint(i)
        if X[0] > SLOT_END + 1e-6:
            continue
        u = displacement.GetTuple3(i)
        key = round(X[0], 6)
        if abs(X[1] - Y0) < 1e-6:
            lower[key] = (X[0] + u[0], X[1] + u[1])
        elif abs(X[1] - Y1) < 1e-6:
            upper[key] = (X[0] + u[0], X[1] + u[1])
    return [lower[k] for k in sorted(lower)], [upper[k] for k in sorted(upper)]


def touching_nodes(lower, upper):
    """Nodes of the lower face that touch or penetrate the upper face (vertical gap <= 0)."""
    result = []
    for x, y in lower:
        for (xa, ya), (xb, yb) in zip(upper[:-1], upper[1:]):
            if xb != xa and min(xa, xb) <= x <= max(xa, xb):
                if ya + (yb - ya) * (x - xa) / (xb - xa) - y <= 1e-12:
                    result.append((x, y))
                break
    return result


def text(string, x, y, size, style="regular", color=BLACK, justification="left",
         coordinates="viewport", orientation=0.0, vertical="bottom"):
    actor = vtk.vtkTextActor()
    actor.SetInput(string)
    prop = actor.GetTextProperty()
    prop.SetFontFamily(vtk.VTK_FONT_FILE)
    prop.SetFontFile(FONT[style])
    prop.SetFontSize(size)
    prop.SetColor(*color)
    prop.SetOrientation(orientation)
    {"left": prop.SetJustificationToLeft, "center": prop.SetJustificationToCentered,
     "right": prop.SetJustificationToRight}[justification]()
    {"bottom": prop.SetVerticalJustificationToBottom, "center": prop.SetVerticalJustificationToCentered,
     "top": prop.SetVerticalJustificationToTop}[vertical]()
    coordinate = actor.GetPositionCoordinate()
    if coordinates == "world":
        coordinate.SetCoordinateSystemToWorld()
        coordinate.SetValue(x, y, 0.0)
    else:
        coordinate.SetCoordinateSystemToNormalizedViewport()
        coordinate.SetValue(x, y)
    return actor


def polyline(points, color, width, z=0.01):
    pts = vtk.vtkPoints()
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(points))
    for i, (x, y) in enumerate(points):
        pts.InsertNextPoint(x, y, z)
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(width)
    actor.GetProperty().SetLighting(False)
    return actor


def markers(points, color, size, z=0.02):
    pts = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    for x, y in points:
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pts.InsertNextPoint(x, y, z))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetVerts(verts)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetPointSize(size)
    actor.GetProperty().SetRenderPointsAsSpheres(True)
    return actor


def view(window, viewport, xmin, xmax, ymin, ymax):
    """Renderer with a parallel camera that shows at least [xmin, xmax] x [ymin, ymax]."""
    renderer = vtk.vtkRenderer()
    renderer.SetViewport(*viewport)
    renderer.SetBackground(1.0, 1.0, 1.0)
    aspect = ((viewport[2] - viewport[0]) * WIDTH) / ((viewport[3] - viewport[1]) * HEIGHT)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half_height = 0.5 * max(ymax - ymin, (xmax - xmin) / aspect)
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetFocalPoint(cx, cy, 0.0)
    camera.SetPosition(cx, cy, 5.0)
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetParallelScale(half_height)
    window.AddRenderer(renderer)
    return renderer


def render(grid, push, n_active, curve, index, path):
    lut = lookup_table()
    warp = vtk.vtkWarpVector()
    warp.SetInputData(grid)
    warp.SetScaleFactor(1.0)
    lower, upper = slot_faces(grid)
    touching = touching_nodes(lower, upper)

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(WIDTH, HEIGHT)
    window.SetMultiSamples(8)

    def body(edges, edge_width):
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(warp.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0.0, MAX_U)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLighting(False)
        if edges:
            actor.GetProperty().SetEdgeVisibility(True)
            actor.GetProperty().SetEdgeColor(0.15, 0.15, 0.15)
            actor.GetProperty().SetLineWidth(edge_width)
        return actor

    # background with the titles
    page = vtk.vtkRenderer()
    page.SetViewport(0.0, 0.0, 1.0, 1.0)
    page.SetBackground(1.0, 1.0, 1.0)
    page.SetLayer(0)
    page.AddActor2D(text("Self-contact of a closing slot", 0.03, 0.935, 40, "bold"))
    page.AddActor2D(text("Frictionless penalty contact between the two faces of one surface: "
                         "Gauss-point-to-segment with a smoothed (G1) master surface, Verlet "
                         "candidate list, St. Venant–Kirchhoff, Q1 elements",
                         0.03, 0.895, 22, "regular", GRAY))
    page.AddActor2D(text("prescribed displacement  %.4f" % push, 0.97, 0.935, 30, "regular", BLACK, "right"))
    page.AddActor2D(text("(a)  Deformed specimen", 0.03, 0.835, 26, "bold"))
    page.AddActor2D(text("(b)  Detail of the slot mouth", 0.375, 0.835, 26, "bold"))
    page.AddActor2D(text("(c)  Contact length", 0.705, 0.835, 26, "bold"))
    window.SetNumberOfLayers(2)
    window.AddRenderer(page)

    # (a) deformed specimen without mesh edges, frame of the detail
    detail = (-0.012, 0.232, 0.40, 0.58)
    full = view(window, (0.02, 0.17, 0.355, 0.82), -0.06, 1.06, -0.04, 1.06)
    full.SetLayer(1)
    full.AddActor(body(False, 0.0))
    full.AddActor(polyline(lower, LOWER, 2.0))
    full.AddActor(polyline(upper, UPPER, 2.0))
    x0, x1, y0, y1 = detail
    full.AddActor(polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], BLACK, 2.0, 0.03))

    # (b) detail with the mesh, the reference position and the contact zone
    zoom = view(window, (0.365, 0.17, 0.69, 0.82), *detail)
    zoom.SetLayer(1)
    zoom.AddActor(body(True, 1.0))
    zoom.AddActor(polyline([(0.0, Y0), (SLOT_END, Y0)], GRAY, 1.5))
    zoom.AddActor(polyline([(0.0, Y1), (SLOT_END, Y1)], GRAY, 1.5))
    zoom.AddActor(polyline(lower, LOWER, 4.0))
    zoom.AddActor(polyline(upper, UPPER, 4.0))
    if touching:
        zoom.AddActor(markers(touching, CONTACT, 11))

    # (c) contact length versus prescribed displacement
    chart = view(window, (0.705, 0.17, 0.985, 0.82), -0.30, 1.06, -0.26, 1.06)
    chart.SetLayer(1)
    length_max = max(c[1] for c in curve)
    y_step = 0.1 if length_max > 0.25 else 0.05
    y_top = y_step * (int(length_max / y_step) + 1)
    sx, sy = 1.0 / PUSH_MAX, 1.0 / y_top
    chart.AddActor(polyline([(0.0, 0.0), (1.0, 0.0)], BLACK, 1.5))
    chart.AddActor(polyline([(0.0, 0.0), (0.0, 1.0)], BLACK, 1.5))
    for d in [0.0, 0.01, 0.02, 0.03, 0.04]:
        chart.AddActor(polyline([(d * sx, 0.0), (d * sx, -0.02)], BLACK, 1.5))
        chart.AddActor2D(text("%.2f" % d, d * sx, -0.045, 20, "regular", BLACK, "center",
                              "world", vertical="top"))
    y = 0.0
    while y <= y_top + 1e-12:
        chart.AddActor(polyline([(0.0, y * sy), (-0.02, y * sy)], BLACK, 1.5))
        chart.AddActor(polyline([(0.0, y * sy), (1.0, y * sy)], (0.88, 0.88, 0.88), 1.0, -0.01))
        chart.AddActor2D(text("%.1f" % y if y_step >= 0.1 else "%.2f" % y, -0.04, y * sy, 20,
                              "regular", BLACK, "right", "world", vertical="center"))
        y += y_step
    chart.AddActor2D(text("prescribed displacement", 0.5, -0.16, 22, "italic", BLACK,
                          "center", "world", vertical="top"))
    chart.AddActor2D(text("contact length", -0.21, 0.5, 22, "italic", BLACK, "center", "world",
                          90.0, "center"))
    chart.AddActor(polyline([(d * sx, l * sy) for d, l in curve], (0.80, 0.80, 0.80), 2.0))
    chart.AddActor(polyline([(d * sx, l * sy) for d, l in curve[:index + 1]], UPPER, 3.5, 0.02))
    chart.AddActor(markers([(curve[index][0] * sx, curve[index][1] * sy)], CONTACT, 14, 0.03))

    # legend and color bar
    legend = page
    legend.AddActor2D(text("lower face", 0.39, 0.105, 21, "regular", LOWER))
    legend.AddActor2D(text("upper face", 0.47, 0.105, 21, "regular", UPPER))
    legend.AddActor2D(text("contact (non-positive gap)", 0.55, 0.105, 21, "regular", CONTACT))
    legend.AddActor2D(text("reference position", 0.39, 0.065, 21, "regular", GRAY))
    legend.AddActor2D(text("active contact points: %d" % n_active, 0.55, 0.065, 21, "regular",
                           BLACK))
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetOrientationToHorizontal()
    bar.SetNumberOfLabels(4)
    bar.SetLabelFormat("%.3f")
    bar.SetTitle("")
    bar.SetBarRatio(0.35)
    bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    bar.GetPositionCoordinate().SetValue(0.03, 0.035)
    bar.SetWidth(0.30)
    bar.SetHeight(0.09)
    bar.SetTextPositionToPrecedeScalarBar()
    for prop in (bar.GetLabelTextProperty(),):
        prop.SetFontFamily(vtk.VTK_FONT_FILE)
        prop.SetFontFile(FONT["regular"])
        prop.SetColor(*BLACK)
        prop.SetItalic(False)
        prop.SetBold(False)
        prop.SetShadow(False)
    bar.SetUnconstrainedFontSize(True)
    bar.GetLabelTextProperty().SetFontSize(20)
    page.AddActor2D(bar)
    page.AddActor2D(text("displacement magnitude |u|", 0.18, 0.135, 21, "italic", BLACK, "center"))

    window.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(window)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    window.Finalize()


states = []
first = sorted(glob.glob(os.path.join(runs_dir, "k001", "vtu", "*_0000.0.vtu")))[0]
states.append((first, 0.0, 0, 0.0))
for k in range(1, N + 1):
    run = os.path.join(runs_dir, "k%03d" % k)
    vtu = sorted(glob.glob(os.path.join(run, "vtu", "*_0001.0.vtu")))[0]
    contact_file = sorted(glob.glob(os.path.join(run, "*_contact_deformable.txt")))[0]
    last = open(contact_file).read().strip().splitlines()[-1].split()
    # the active area counts both faces of the slot (self-contact): contact length = area / 2
    states.append((vtu, PUSH_MAX * k / N, int(last[3]), 0.5 * float(last[4])))

curve = [(push, length) for _, push, _, length in states]
for index, (vtu, push, n_active, _) in enumerate(states):
    render(read(vtu), push, n_active, curve, index, os.path.join(frames_dir, "frame_%03d.png" % index))
print("frames:", len(states))
