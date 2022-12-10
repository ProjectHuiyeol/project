/**
 * Template Name: Knight - v2.2.1
 * Template URL: https://bootstrapmade.com/knight-free-bootstrap-theme/
 * Author: BootstrapMade.com
 * License: https://bootstrapmade.com/license/
 */
!(function ($) {
  'use strict'
  // drag & drop
  // 1st tag -> playlist
  const columns_1st = document.querySelectorAll('.column_1st_tag')
  const columns_2nd = document.querySelectorAll('.column_2nd_tag')
  const columns_playlist = document.querySelectorAll('.column_playlist')
  const columns_discard = document.querySelectorAll('.trash_can')

  columns_discard.forEach((column) => {
    new Sortable(column, {
      group: {
        name: 'shared',
        pull: 'false'
      }
    })
  })

  // columns_discard.forEach
  columns_1st.forEach((column) => {
    new Sortable(column, {
      group: {
        name: 'shared',
        pull: 'clone',
        put: false
      },
      // multiDrag : true,
      animation: 150,
      sort: false
    })
  })
  columns_2nd.forEach((column) => {
    new Sortable(column, {
      group: {
        name: 'shared',
        pull: 'clone',
        put: false
      },
      // multiDrag : true,
      animation: 150,
      sort: false
    })
  })
  columns_playlist.forEach((column) => {
    const sort_column = new Sortable(column, {
      group: {
        name: 'shared'
      },
      animation: 150
    })
  })

  $('#add_playlist_btn').click(function () {
    const selected_song = document.querySelector('.column_playlist')
    const input_songs = document.querySelector('#song_names')
    // console.log(selected_song.childElementCount);
    // console.log(selected_song.children[0])
    var song_list = []
    for (let i = 0; i < selected_song.childElementCount; i++) {
      song_list.push(selected_song.children[i].textContent)
    }
    input_songs.value = song_list
    console.log(song_list)
  })
})(jQuery)
