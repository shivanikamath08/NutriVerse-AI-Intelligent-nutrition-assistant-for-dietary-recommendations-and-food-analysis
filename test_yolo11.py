from ultralytics import YOLO

model = YOLO("yolo11n.pt")



# Run detection on an image
results = model.predict(source="d.png", show=True)
