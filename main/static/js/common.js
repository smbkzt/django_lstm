$('#ajax-form').on('submit', function(event){
    event.preventDefault();
    $(".loader").css("display", "block");
    $("#answer").css("display", "none");
    $.ajax({
        url : "/",
        type : "POST",
        data : { "the_origin" : $('#input-origin-message').val(),
                 "the_comment": $('#input-comment-message').val() },
        success : function(json) {
            $(".loader").css("display", "none");
            $("#answer").css("display", "block");
            if (json == "The comment message has agreement sentiment"){
                $("#answer").html('<div class="alert alert-success">' + json + '</div>');
            }
            else{
                $("#answer").html('<div class="alert alert-danger">' + json + '</div>');
            }
        }

    });
});
