@echo off
setlocal
if not exist build mkdir build
pushd build
cl.exe /nologo /EHsc /std:c++17 /LD /O2 /MT /W4 /Brepro /DWIN32 ../clipper.cpp ../ppocr_db_jt_round.cpp /link /OUT:ppocr_db_jt_round.dll
set result=%errorlevel%
popd
exit /b %result%
