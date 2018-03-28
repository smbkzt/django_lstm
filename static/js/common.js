$('#ajax-lstm-form').on('submit', function(event){
    event.preventDefault();
    $(".loader").css("display", "inline-block");
    $("#answer").css("display", "none");
    $.ajax({
        url : "/try-lstm",
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

$('#ajax-tweet-form').on('submit', function(event){
    event.preventDefault();
    $(".loader").css("display", "inline-block");
    $("#answer").css("display", "inline-block");
    $.ajax({
        url : "/tweet-search",
        type : "POST",
        data : { "keywords" : $('#input-keyword').val()},
        success : function(json) {
            var data = JSON.stringify(json);
            $(".loader").css("display", "none");
            $("#answer").css("display", "none");
            var parsed = JSON.parse(data);
            var container = document.getElementById('mynetwork');
            var data = {
                nodes: parsed.nodes,
                edges: parsed.edges
            };
            var options = {
                edges:{
                    arrows: {
                        from: {enabled: true, scaleFactor:0.75, type:'arrow'}
                    },
                    length: 150,
                },
                nodes:{
                    brokenImage: undefined,
                    shape: 'circularImage',
                    chosen: true,
                }
            }
            var network = new vis.Network(container, data, options);
        }

    });
});
