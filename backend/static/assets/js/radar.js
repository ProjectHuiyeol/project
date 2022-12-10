!(function($){
  $(document).ready(function(){
    // sound_score
    var sound_happy = document.querySelector('#sound_sentiment0').value;
    var sound_sad = document.querySelector('#sound_sentiment1').value;
    var sound_angry = document.querySelector('#sound_sentiment2').value;
    var sound_relaxed = document.querySelector('#sound_sentiment3').value;
    // lyrics_score
    var lyrics_happy = document.querySelector('#lyrics_sentiment0').value;
    var lyrics_fear = document.querySelector('#lyrics_sentiment1').value;
    var lyrics_angry = document.querySelector('#lyrics_sentiment2').value;
    var lyrics_dislike = document.querySelector('#lyrics_sentiment3').value;
    var lyrics_surprise = document.querySelector('#lyrics_sentiment4').value;
    var lyrics_sad = document.querySelector('#lyrics_sentiment5').value;
    
    // sound_color
    var red_sound_calc = parseFloat(253*sound_happy +  179*sound_angry  + 82*sound_sad + 82*sound_relaxed);
    var green_sound_calc = parseFloat(251*sound_happy +  6*sound_angry  + 83*sound_sad + 252*sound_relaxed);
    var blue_sound_calc = parseFloat(84*sound_happy +  7*sound_angry + 225*sound_sad + 81*sound_relaxed);

    // lyrics_color
    var red_lyrics_calc = parseFloat(253*lyrics_happy + 3*lyrics_fear + 179*lyrics_angry + 224*lyrics_dislike + 91*lyrics_surprise + 82*lyrics_sad);
    var green_lyrics_calc = parseFloat(251*lyrics_happy + 149*lyrics_fear + 6*lyrics_angry + 89*lyrics_dislike + 190*lyrics_surprise + 83*lyrics_sad);
    var blue_lyrics_calc = parseFloat(84*lyrics_happy + 6*lyrics_fear + 7*lyrics_angry + 232*lyrics_dislike + 255*lyrics_surprise + 225*lyrics_sad);
    
    // sound_chart_section
    var ctx = $("#radar-chart1");
    // sound chart drawing
    var sound_radarChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ["HAPPY", "SAD", "ANGRY", "RELAXED"],
        datasets: [
          {
            label: "SOUND",
            fill: true,
            backgroundColor: "rgba("+red_sound_calc+","+green_sound_calc+","+blue_sound_calc+",0.8)",
            borderColor: "rgba("+red_sound_calc+","+green_sound_calc+","+blue_sound_calc+",1)",
            pointBorderColor: "#fff",
            pointBackgroundColor: "rgba("+red_sound_calc+","+green_sound_calc+","+blue_sound_calc+",1)",
            data: [sound_happy,sound_sad,sound_angry,sound_relaxed]
          }
        ]
      },
      options: {
        title: {
          display: true,
          text: 'SOUND SENTIMENT OF PLAYLIST'
        }
      }
    });
    var lyrics = $("#radar-chart2");
    var lyrics_radarChart = new Chart(lyrics, {
      type: 'radar',
      data: {
        labels: ["HAPPY", "FEAR", "ANGRY", "DISLIKE", "SURPRISE", "SAD"],
        datasets: [
          {
            label: "LYRICS",
            fill: true,
            backgroundColor: "rgba("+red_lyrics_calc+","+green_lyrics_calc+","+blue_lyrics_calc+",0.8)",
            borderColor: "rgba("+red_lyrics_calc+","+green_lyrics_calc+","+blue_lyrics_calc+",0.8)",
            pointBorderColor: "#fff",
            pointBackgroundColor: "rgba("+red_lyrics_calc+","+green_lyrics_calc+","+blue_lyrics_calc+",0.8)",
            data: [lyrics_happy,lyrics_fear,lyrics_angry,lyrics_dislike,lyrics_surprise,lyrics_sad]
          }
        ]
      },
      options: {
        title: {
          display: true,
          text: 'LYRICS SENTIMENT OF PLAYLIST'
        }
      }
    });
  });
})(jQuery);
