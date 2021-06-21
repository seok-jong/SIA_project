FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev wget vim

RUN pip install keras numpy pillow flask gunicorn opencv-python segmentation_models albumentations

WORKDIR root
COPY ./app.py ./app.py
COPY ./templates/index.html ./templates/index.html
COPY ./images/sample_image.png ./images/sample_image.png

RUN mkdir model

COPY ./model/mixed_928.h5 ./model/mixed_928.h5

EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"]
