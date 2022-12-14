!(function($){
    $(document).ready(function(){
        LoadingWithMask('../img/Spinner.gif');
        setTimeout(closeLoadingWithMask(),600000);
    });
    function LoadingWithMask(gif){
        // 화면의 높이와 너비 구하기
        var maskHeight = $(document).height();
        var maskWidth = window.document.body.clientWidth;
    
        // 화면에 출력할 마스크 설정
        var mask = "<div id='mask' style='position:absolute; z-index:9000; background-color:#000000; display:none; left:0; top:0'></div>";
        var loadingImg = '';
    
        loadingImg += "<img src='" + gif + "' style='position: absolute; display: block; margin : 0px auto;'/>";
    
        // 화면에 레이어 추가
        $('body').append(mask)
    
        //마스크의 높이와 너비를 화면 것으로 만들어 전체 화면 채우기
        $('#mask').css({
          'width': maskWidth,
          'height': maskHeight,
          'opacity': '0.3'
        });
    
        //마스크 표시
        $('#mask').show()
        //로딩중 이미지 표시
        $('#loadingImg').append(loadingImg);
        $('#loadingImg').show();
    }
    
    function closeLoadingWithMask() {
        $('#mask, #loadingImg').hide();
        console.log("오오오오")
        $('#mask, #loadingImg').empty();  
    }
})(jQuery);