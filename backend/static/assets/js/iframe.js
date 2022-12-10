// document.getElementById("goiframe").onclick = function () {
//     alert("Hello World");
// };

function postYourAdd () {
    var iframe = $("#forPostyouradd");
    iframe.attr("src", iframe.data("src")); 
}
 
$("button").on("click", postYourAdd);