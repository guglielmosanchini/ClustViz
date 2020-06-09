# Install script for directory: /Users/ity9dw37/Desktop/metis-5.1.0/programs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/programs/gpmetis")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gpmetis" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gpmetis")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/libmetis/libmetis.dylib" "libmetis.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gpmetis")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/home/karypis/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gpmetis")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gpmetis")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/programs/ndmetis")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/ndmetis" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/ndmetis")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/libmetis/libmetis.dylib" "libmetis.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/ndmetis")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/home/karypis/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/ndmetis")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/ndmetis")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/programs/mpmetis")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/mpmetis" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/mpmetis")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/libmetis/libmetis.dylib" "libmetis.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/mpmetis")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/home/karypis/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/mpmetis")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/mpmetis")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/programs/m2gmetis")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/m2gmetis" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/m2gmetis")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/libmetis/libmetis.dylib" "libmetis.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/m2gmetis")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/home/karypis/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/m2gmetis")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/m2gmetis")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/programs/graphchk")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/graphchk" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/graphchk")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/libmetis/libmetis.dylib" "libmetis.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/graphchk")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/home/karypis/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/graphchk")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/graphchk")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/programs/cmpfillin")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cmpfillin" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cmpfillin")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -change "/Users/ity9dw37/Desktop/metis-5.1.0/build/Darwin-x86_64/libmetis/libmetis.dylib" "libmetis.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cmpfillin")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/home/karypis/local/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cmpfillin")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/cmpfillin")
    endif()
  endif()
endif()

