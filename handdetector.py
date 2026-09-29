import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_threshold=0.5,
                 tracking_threshold=0.5, point_color=(0,0,255), connections_color=(255,255,255), ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_threshold = detection_threshold
        self.tracking_threshold = tracking_threshold
        self.point_color = point_color
        self.connections_color = connections_color

        self.hands_mp = mp.solutions.hands
        self.hands = self.hands_mp.Hands(
            self.mode,
            self.max_hands,
            1,
            self.detection_threshold,
            self.tracking_threshold,
        )

        self.drawing_mp = mp.solutions.drawing_utils

        self.drawing_points_config = self.drawing_mp.DrawingSpec(color=self.point_color)
        self.drawing_connections_config = self.drawing_mp.DrawingSpec(color=self.connections_color)

    def find_hands(self, image, drawing=True):
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_RGB)
        if self.results.multi_hand_landmarks:
            for points in self.results.multi_hand_landmarks:
                if drawing:
                    self.drawing_mp.draw_landmarks(
                        image, points, self.hands_mp.HAND_CONNECTIONS, self.drawing_points_config, self.drawing_connections_config
                    )
        return image

    def find_points(self, image, hand_num=0, drawing=True, color=(255,0,0), radius=7):
        point_list = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            for id, point in enumerate(hand.landmark):
                height, width, _ = image.shape
                center_x, center_y = int(point.x * width), int(point.y * height)
                point_list.append([id, center_x, center_y])

        return point_list

def main():
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        _, image = cap.read()
        image = cv2.flip(image, 1)
        image = detector.find_hands(image)
        point_list = detector.find_points(image)
        print(point_list)
        cv2.imshow('Captura', image)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()