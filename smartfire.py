from ultralytics import YOLO
import cv2


# ==========================================
# LOAD MODEL YOLO
# ==========================================

model = YOLO("best.pt")


# ==========================================
# THRESHOLD HASIL GENETIC ALGORITHM
# ==========================================

FIRE_THRESHOLD = 0.708
SMOKE_THRESHOLD = 0.604


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Kamera tidak dapat dibuka!")

    exit()


print("====================================")
print("       SMARTFIRE AI")
print("       YOLO + GENETIC ALGORITHM")
print("====================================")

print(
    f"Fire Threshold  : {FIRE_THRESHOLD}"
)

print(
    f"Smoke Threshold : {SMOKE_THRESHOLD}"
)

print("Tekan Q untuk keluar")


# ==========================================
# LOOP CAMERA
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Gagal membaca kamera!")

        break


    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model(
        frame,
        verbose=False
    )


    annotated_frame = results[0].plot()


    fire_detected = False
    smoke_detected = False

    highest_fire = 0
    highest_smoke = 0


    # ======================================
    # BACA HASIL YOLO
    # ======================================

    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[
                class_id
            ].lower()


            # ==============================
            # FIRE
            # ==============================

            if class_name == "fire":

                highest_fire = max(
                    highest_fire,
                    confidence
                )

                if confidence >= FIRE_THRESHOLD:

                    fire_detected = True


            # ==============================
            # SMOKE
            # ==============================

            elif class_name == "smoke":

                highest_smoke = max(
                    highest_smoke,
                    confidence
                )

                if confidence >= SMOKE_THRESHOLD:

                    smoke_detected = True


    # ======================================
    # KEPUTUSAN
    # ======================================

    if fire_detected:

        status = "BAHAYA - API TERDETEKSI"

    elif smoke_detected:

        status = "PERINGATAN - ASAP TERDETEKSI"

    else:

        status = "AMAN"


    # ======================================
    # TAMPILKAN STATUS
    # ======================================

    cv2.putText(
        annotated_frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    # ======================================
    # TAMPILKAN THRESHOLD
    # ======================================

    cv2.putText(
        annotated_frame,
        f"Fire Threshold: {FIRE_THRESHOLD:.3f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Smoke Threshold: {SMOKE_THRESHOLD:.3f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ======================================
    # TAMPILKAN CONFIDENCE
    # ======================================

    if highest_fire > 0:

        cv2.putText(
            annotated_frame,
            f"Fire: {highest_fire:.2f}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    if highest_smoke > 0:

        cv2.putText(
            annotated_frame,
            f"Smoke: {highest_smoke:.2f}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ======================================
    # TAMPILKAN CAMERA
    # ======================================

    cv2.imshow(
        "SmartFire AI - YOLO + Genetic Algorithm",
        annotated_frame
    )


    # ======================================
    # EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()