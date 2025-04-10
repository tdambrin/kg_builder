from scholarly import scholarly  # type: ignore

authors = [
    next(scholarly.search_author(name))
    for name in [
        "Patrick Lewis",
        "Ethan Perez",
        "Aleksandra Piktus",
        "Fabio Petroni",
        "Vladimir Karpukhin",
        "Naman Goyal",
        "Heinrich Küttler",
        "Mike Lewis",
        "Wen-tau Yih",
        "Tim Rocktäschel",
        "Sebastian Riedel",
        "Douwe Kiela",
    ]
]

print(authors)
