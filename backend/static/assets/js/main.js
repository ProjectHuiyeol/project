/**
* Template Name: Knight - v2.2.1
* Template URL: https://bootstrapmade.com/knight-free-bootstrap-theme/
* Author: BootstrapMade.com
* License: https://bootstrapmade.com/license/
*/
!(function($) {
  "use strict";
  // drag & drop
  // 1st tag -> playlist
  const columns_1st = document.querySelectorAll(".column_1st_tag");
  // const columns_2nd = document.querySelectorAll(".column_2nd_tag");
  const columns_playlist = document.querySelectorAll(".column_playlist");
  

  columns_1st.forEach((column) => {
    const sort_column = new Sortable(column, {
      group: {
        name : "shared",
        pull : 'clone',
        put : false
      },
      // multiDrag : true,
      animation: 150,
      sort : false
    });
  });
  
  // columns_2nd.forEach((column) => {
  //   new Sortable(column, {
  //     group: {
  //       name : "shared",
  //       pull : 'clone',
  //       put : false
  //     },
  //     // multiDrag : true,
  //     animation: 150,
  //     sort : false
  //   });
  // });
  
  columns_playlist.forEach((column) => {
    new Sortable(column, {
      group: {
        name : "shared"
      },
      animation: 150
    });
  });
  $('#remove_song').click(function(){
    console.log('nono');
  });
  $('#add_playlist_btn').click(function() {
    
    const selected_song = document.querySelector('.column_playlist');
    const input_songs = document.querySelector('#song_names');
    // console.log(selected_song.childElementCount);
    // console.log(selected_song.children[0])
    var song_list = [];
    for (let i = 0 ; i<selected_song.childElementCount; i++){
      song_list.push(selected_song.children[i].textContent);
    }
    input_songs.value = song_list;
    console.log(song_list);
    LoadingWithMask("../static/assets/img/loading/loading.gif");
    setTimeout(closeLoadingWithMask,30000)

  });

  function LoadingWithMask(gif){
    // 화면의 높이와 너비 구하기
    var maskHeight = $(document).height();
    var maskWidth = window.document.body.clientWidth;

    // 화면에 출력할 마스크 설정
    var mask = "<div id='mask' style='position:absolute; z-index:9000; background-color:#2893EF; display:none; left:0; top:0'></div>";
    var loadingImg = '';
    console.log(gif);

    loadingImg += "<div id='loadingImg'>";
    loadingImg += " <img name = 'img_check' src='" + gif + "' style='display: block; margin :0px auto; width:100%; height:100%; '/>";
    loadingImg += "</div>";

    // 화면에 레이어 추가
    $('body').append(mask)
    $('#mask').append(loadingImg)

    //마스크의 높이와 너비를 화면 것으로 만들어 전체 화면 채우기
    $('#mask').css({
    'width': maskWidth,
    'height': maskHeight,
    'opacity': '1'
    });

    //마스크 표시
    $('#mask').show()
    //로딩중 이미지 표시
    $('#loadingImg').show();
}
function closeLoadingImg(){
    $('#loadingImg').hide();
    $('#loadingImg').empty();
}
function closeLoadingWithMask() {
    $('#mask, #loadingImg').hide();
    $('#mask, #loadingImg').empty();
    // $('#mask, #loadingImg').remove();  
}
  
})(jQuery);