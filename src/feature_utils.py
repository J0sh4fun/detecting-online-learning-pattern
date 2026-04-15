import math

def calculate_distance(p1, p2):
    """
    Calculates the Euclidean distance between two 2D points.

    Args:
        p1 (tuple): The (x, y) coordinates of the first point.
        p2 (tuple): The (x, y) coordinates of the second point.

    Returns:
        float: The distance between p1 and p2.
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_midpoint(p1, p2):
    """
    Calculates the midpoint between two 2D points.

    Args:
        p1 (tuple): The (x, y) coordinates of the first point.
        p2 (tuple): The (x, y) coordinates of the second point.

    Returns:
        tuple: The (x, y) coordinates of the midpoint.
    """
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

def calculate_vertical_angle(p_top, p_bottom):
    """
    Tính góc lệch của đoạn thẳng (từ p_bottom đến p_top) so với trục dọc.
    Trục dọc thẳng đứng = 0 độ.
    """
    dx = p_top[0] - p_bottom[0]
    dy = p_bottom[1] - p_top[1] # Trục y của ảnh hướng xuống dưới, nên cần đảo ngược
    
    # Tính góc bằng radian sau đó chuyển sang độ
    # Thêm abs(dx) để luôn lấy góc nhọn lệch khỏi trục dọc (trái hay phải đều dương)
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)