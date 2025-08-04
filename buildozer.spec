[app]
title = PhotoGeo
package.name = photo_geo
package.domain = org.example

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,txt

version = 0.1
requirements = python3,kivy==2.1.0
android.permissions = CAMERA,ACCESS_FINE_LOCATION
android.api = 31
android.minapi = 21
android.ndk = 25b
android.arch = arm64-v8a

[buildozer]
log_level = 2
android.accept_sdk_license = True
