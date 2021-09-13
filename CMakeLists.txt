cmake_minimum_required(VERSION 3.19)
project(Kann VERSION 0.0.0 LANGUAGES C)
string(TOLOWER "${PROJECT_NAME}" PROJECT_LOWER_NAME)

set(CMAKE_C_STANDARD 90)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
if (BUILD_SHARED_LIBS)
    set(LIBRARY_TYPE_FLAG "SHARED")
else ()
    set(LIBRARY_TYPE_FLAG "STATIC")
endif ()

# control where the static and shared libraries are built so that on windows
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_library("${PROJECT_LOWER_NAME}_compiler_flags" INTERFACE)
target_compile_features("${PROJECT_LOWER_NAME}_compiler_flags" INTERFACE "c_std_90")

set(gcc_like "$<COMPILE_LANG_AND_ID:C,CXX,ARMClang,AppleClang,Clang,GNU>")
set(msvc "$<COMPILE_LANG_AND_ID:C,CXX,MSVC>")
# -g -fsanitize=address -fno-omit-frame-pointer
target_compile_options(
        "${PROJECT_LOWER_NAME}_compiler_flags"
        INTERFACE
        "$<${gcc_like}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
        "$<${msvc}:$<BUILD_INTERFACE:-W3;-WX;-Zi;-permissive->>"
)
if (CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

add_subdirectory("kann")
add_subdirectory("kann_extra")

# configure a header file to pass the version number only
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.h.in"
        "${PROJECT_NAME}Config.h"
)

install(
        FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.h"
        DESTINATION "include"
)

include(InstallRequiredSystemLibraries)
set(CPACK_BUNDLE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VENDOR "SamuelMarks")
set(CPACK_PACKAGE_DESCRIPTION "Very simple HTTPS interface built atop libcurl.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Very simple HTTPS interface built atop libcurl.")
if (APPLE)
    set(CPACK_BUNDLE_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Info.plist")
    set(CPACK_BUNDLE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Info.plist")
    set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CustomVolumeIcon.icns")
endif()
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${${PROJECT_NAME}_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${${PROJECT_NAME}_VERSION_MINOR}")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/cmake/README.txt")
set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Welcome.txt")
set(CPACK_PACKAGE_CONTACT "https://github.com/attractivechaos/kann")

include(CPack)
include(CMakePackageConfigHelpers)

# install the configuration targets
install(
        EXPORT "kann_extraTargets"
        FILE "kann_extraTargets.cmake"
        DESTINATION "lib/cmake/kann_extra"
)
install(
        EXPORT "kannTargets"
        FILE "kannTargets.cmake"
        DESTINATION "lib/cmake/kann"
)

# generate the config file that is includes the exports
configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/kann_extraConfig.cmake"
        INSTALL_DESTINATION "lib/cmake/example"
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# generate the version file for the config file
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/kann_extraConfigVersion.cmake"
        VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}"
        COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(
        FILES
        "${CMAKE_CURRENT_BINARY_DIR}/KannConfig.cmake"
        DESTINATION "lib/cmake/kann_extra"
)

# generate the export targets for the build tree
# needs to be after the install(TARGETS ) command
export(
        EXPORT "kannTargets"
        FILE "${CMAKE_CURRENT_BINARY_DIR}/kannTargets.cmake"
)
export(
        EXPORT "kann_extraTargets"
        FILE "${CMAKE_CURRENT_BINARY_DIR}/kann_extraTargets.cmake"
)
