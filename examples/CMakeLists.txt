set(examples "ae" "inspect" "mlp" "mnist-cnn" "rnn-bit" "textgen" "vae")
foreach(example ${examples})
    set(EXEC_NAME "kann_example_${example}")
    set(Source_Files "${example}.c")
    source_group("${EXEC_NAME} Source Files" FILES "${Source_Files}")

    add_executable("${EXEC_NAME}" "${Source_Files}")

    set(_libs "kann")
    if (example STREQUAL "minst-cnn")
        list(APPEND _libs "kann_extra")
    endif ()
    target_link_libraries("${EXEC_NAME}" PRIVATE "${_libs}")

    set_target_properties(
            "${EXEC_NAME}"
            PROPERTIES
            LINKER_LANGUAGE
            C
    )

    # install rules
    set(installable_libs "${EXEC_NAME}" "${PROJECT_LOWER_NAME}_compiler_flags")
    if (TARGET "${DEPENDANT_LIBRARY}")
        list(APPEND installable_libs "${DEPENDANT_LIBRARY}")
    endif ()
    install(TARGETS ${installable_libs}
            DESTINATION "bin"
            EXPORT "${EXEC_NAME}Targets")
endforeach()
