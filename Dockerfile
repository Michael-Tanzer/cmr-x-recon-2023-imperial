# Specify the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /

# Copy the example directory to /app
COPY . .

# install dependencies from requirements.txt
RUN rm ~/.cache/pip -rf
RUN pip cache purge
RUN pip install --no-cache-dir --upgrade pip==22.3.1
RUN pip install --no-cache-dir --user -r requirements.txt

# DO NOT EDIT THE FLLOWING LINES
COPY *_run.py /
COPY submitter.json /
# You can run more commands when the container start by 
# editing docker-entrypoint.sh
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]