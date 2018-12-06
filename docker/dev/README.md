# Develop with Theia

To develop qcor using the Eclipse Theia IDE and Docker

```bash
$ git clone --recursive https://code.ornl.gov/qci/qcor
$ cd qcor/docker/dev
$ docker-compose up -d
```

Navigate to `http://localhost:3000` in your web browser. 

For an application look and feel in Google Chrome, you can run 
```bash
$ /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --app=http://localhost:3000 (On a Mac)
$ google-chrome --app=http://localhost:3000 (On Linux)
```

To delete this development workspace
```bash
$ docker-compose down
```
