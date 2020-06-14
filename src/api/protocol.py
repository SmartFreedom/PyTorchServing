{
    "prediction":
    {
        "density":
        {
            "response":
            {
                "ab": 0.008354410529136658,
                "cd": 0.9916439652442932
            },
            "threshold": 0.5,
            "argmax": true
        },
        "asymmetry":
        {
            "response":
            {
                "no": 0.6030043403,
                "local": 0.033554545,
                "total": 0.0083544105,
                "local_calc": 0.0083544105,
                "dynamic": 0.0083544105
            },
            "default": "no",
            "threshold": 0.5,
            "argmax": true
        },
        "calcifications":
        {
            "response":
            {
                "no": 0.6030043403,
                "benign": 0.033554545,
                "malignant": 0.0083544105,
                "extra_focal": 0.0083544105
            },
            "default": "no",
            "threshold": 0.5,
            "argmax": true
        },
        "distortions":
        {
            "response":
            {
                "yes": 0.008354410529136658,
                "no": 0.9916439652442932
            },
            "threshold": 0.5,
            "default": "no",
            "argmax": true
        },
        "mass":
        {
            "response":
            {
                "no": 0.6030043403,
                "homogen": 0.033554545,
                "inhomogen": 0.0083544105,
                "circ_margin": 0.0083544105,
                "obscure_margin": 0.0083544105,
                "spiculate_margin": 0.6030043403,
                "regular_shape": 0.033554545,
                "irregular_shape": 0.0083544105,
                "without_calc": 0.0083544105,
                "with_calc": 0.0083544105
            },
            "default": "no",
            "threshold": 0.5,
            "argmax": true
        },
    }
    "paths":
    {
        "sides": {
            1: {
                "calcification": "media/$key/$side/5e473.png",
                "general_binary_mask": "media/$key/$side/1a2.png",
                "structure": "media/$key/$side/1q3.png",
                "border": "media/$key/$side/5e9485b.png",
                "shape": "media/$key/$side/3d2.png",
                "calcification_malignancy": "media/$key/$side/dcscvf02.png",
                "local_structure_disturtion": "media/$key/$side/7ds3a.png",
            },
        }        
    },
    "findings": [
        {
            "key": 4,
            "prob": 0.7960212832711222,
            "type": "calcification",
            "image": 1,
            "geometry":
            {
                "ellipses": [
                {
                    "center":
                    {
                        "x": 300,
                        "y": 30
                    },
                    "radius_x": 70.0,
                    "radius_y": 40.0,
                    "rotation": 0.0
                }]
            }
        }
    ]
}