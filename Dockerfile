FROM tensorflow/tensorflow:2.2.3-gpu

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev wget vim python3.7

RUN pip install keras numpy pillow flask gunicorn opencv-python segmentation_models albumentations

WORKDIR root
COPY ./app.py ./app.py
COPY ./templates/index.html ./templates/index.html
COPY ./images/sample_image.png ./images/sample_image.png

RUN mkdir model
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12WOVC6BdVDuPZZqc8PSk58yczn83zZ0-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12WOVC6BdVDuPZZqc8PSk58yczn83zZ0-" -O mixed_928.h5 && rm -rf /tmp/cookies.txt
RUN mv mixed_928.h5 model/

EXPOSE 80
ENTRYPOINT ["python"]
CMD ["app.py"]
