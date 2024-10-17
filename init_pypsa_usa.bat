@echo off

set "templates=workflow\repo_data\config"
set "destination=workflow\config"
set "existing_files="

for /f "delims=" %%i in ('dir /b "%destination%" ^| findstr /v ".gitkeep"') do (
    set "existing_files=%%i"
    goto :files_found
)

:copy_files
echo Copying config files from "%templates%" to "%destination%"...
xcopy "%templates%\*" "%destination%\" /s /e /y
goto :end

:files_found
echo Existing config files found in "%destination%". Delete the following files and rerun.
dir /b "%destination%" | findstr /v ".gitkeep"

:end