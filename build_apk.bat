@echo off
echo Building APK with Docker...
docker run --rm -v "%cd%:/app" kivy/buildozer bash -c "cd /app && buildozer init && buildozer android debug"
echo APK should be in bin/ folder
pause