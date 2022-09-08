RECT_CORNER_OFFSETS = {
    'MEDIUM': 7.899999999949614e-05
}

def crop_intersection_SimRa(group, start_rect_coords: Tuple[float], end_rect_coords: Tuple[float]):
    start_rect = box(*start_rect_coords)
    end_rect = box(*end_rect_coords)

    mask_first = group.coords.apply(lambda coords: start_rect.contains(Point(coords)))
    mask_end = group.coords.apply(lambda coords: end_rect.contains(Point(coords)))
    
    if any(mask_first) and any(mask_end):
        first = mask_first[mask_first==True].index[0]
        last = mask_end[mask_end==True].index[-1]
        return group.loc0[first:last]
    else:
        # print(f"No path intersections found for filename: '{group.filename.values[0]}'")
        return None


def crop_intersection_bigbox(group, big_box_coords: Tuple[float]):
    big_box = box(*big_box_coords)

    mask = group.coords.apply(lambda coords: big_box.contains(Point(coords)))

    if any(mask):
        masked = mask[mask == True]
        return group.loc0[masked.index[0]:masked.index[-1]]
    
    return None

def get_rect_coords_from_center_point(point_coords: Tuple[float], rect_size: str = 'MEDIUM') -> Tuple[float]:
    try:
        assert len(point_coords) == 2
    except AssertionError:
        print(f"Rectangle needs 4 coordinates, {len(point_coords)} given.")
    
    try:
        offset = RECT_CORNER_OFFSETS[rect_size]
    except KeyError:
        print(f"'rect_size' must be one of {list(RECT_CORNER_OFFSETS.keys())}")

    if point_coords[0] < point_coords[1]:
        print(f"Point {point_coords} interpreted as (longitude, latitude)")
    elif point_coords[0] > point_coords[1]:
        print(f"Point {point_coords} interpreted as (latitude, longitude)")
        point_coords = (point_coords[1], point_coords[0])
    
    return (point_coords[0] - offset, point_coords[1] - offset, point_coords[0] + offset, point_coords[1] + offset)


def get_corner_offset_from_rect_coords(rect_coords: Tuple[float]) -> float:
    try:
        assert len(rect_coords) == 4
    except AssertionError:
        print(f"Rectangle needs 4 coordinates, {len(rect_coords)} given.")
    
    return box(*rect_coords).length / 8