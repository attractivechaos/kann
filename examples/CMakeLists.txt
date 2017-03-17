add_executable( mlp mlp.c )
target_include_directories( mlp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ )
target_link_libraries( mlp kann kann_extra )


add_executable( mnist-cnn mnist-cnn.c )
target_include_directories( mnist-cnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ )
target_link_libraries( mnist-cnn kann kann_extra )

add_executable( inspect inspect.c )
target_include_directories( inspect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ )
target_link_libraries( inspect kann )

add_executable( textgen textgen.c )
target_include_directories( textgen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ )
target_link_libraries( textgen kann )

add_executable( rnn-bit rnn-bit.c )
target_include_directories( rnn-bit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ )
target_link_libraries( rnn-bit kann )

add_executable( ae ae.c )
target_include_directories( ae PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ )
target_link_libraries( ae kann kann_extra )

add_executable( vae vae.c ) 
target_include_directories( vae PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ )
target_link_libraries( vae kann kann_extra )

# install 
install( TARGETS mlp mnist-cnn inspect textgen rnn-bit ae vae DESTINATION bin )
