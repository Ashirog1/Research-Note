#pylint: skip-file
"""
Predefined graphs for testing and experimentation.

* Graphs are defined via lists containing tuples
* Each tuple in the list represents an edge
* Edges are defined as (start, end, length)

For example, an equilateral triangle like:

   1
  / \
 2 - 3

Would be represented as:

triangle = [(1, 2, 1), (2, 3, 1), (3, 1 ,1)]

"""

# Non-Eularian, single edge, 1-2
single = [(1,2,1)]

# Eularian, simple square, 1-2-3-4
square = [(1,2,1), (2,3,1), (3,4,1), (4,1,1)]

# Semi-Eularian, 2 triangles
ice_cream = [(1,2,4), (1,3,3), (1,4,5), (2,3,3), (3,4,5)]

# Non-Eularian, 3 triangles
sailboat = [
    (1,2,4), (1,3,3), (1,5,10), (2,3,2), (2,4,3), (3,4,3), (4,5,9)
]

# Semi-Eularian, 2 triangles w/ a tail
kite = [(1,2,4), (2,3,3), (3,4,2), (2,4,3), (5,4,2), (4,1,3)]

# Eularian, square w/ parallel edges
clover = [
    (1,2,1), (1,2,2), (2,3,1), (2,3,2), (3,4,1), (3,4,2), (4,1,1), (4,1,2)
]

# Non-eularian w/ 6 odd nodes
big_six = [
    (1,2,8), (1,5,4), (1,8,3), (2,3,9), (2,7,6), (3,4,5),
    (3,6,3), (4,5,5), (4,6,1), (5,6,2), (5,7,3), (7,8,1),
]

# Non-Eularian, 5 adjacent squares
ladder = [
    (1, 2,1), ( 1,12,1), ( 2, 3,1), (2,11,1), (3,4,1), (3,10,1),
    (4, 5,1), ( 4, 9,1), ( 5, 6,1), (5, 8,1), (6,7,1), (7, 8,1), (8,9,1),
    (9,10,1), (10,11,1), (11,12,1)
]

# North of University Ave to beaches
north = [
    ( 1, 2, 1), ( 2, 3, 1), ( 3, 4, 1), ( 3,25, 6), ( 2, 4, 1), ( 2,24, 7),
    ( 4, 5, 3), ( 5, 6, 1), ( 5,26, 6), ( 6, 7, 1), ( 6, 8, 1), ( 8, 9,12),
    ( 8,26, 7), ( 9,10, 5), ( 9,19, 4), (10,11, 1), (10,19, 1), (10,12, 2),
    (12,13, 4), (12,18, 1), (13,14, 1), (13,15, 1), (13,16, 4),
    (17,18, 8), (18,19, 1), (18,20, 1),
    (19,20, 1), (20,21, 5), (21,22, 1), (22,23, 1), (22,24, 5), (21,25,4),
    (24,25, 3), (25,26, 1), (26,27, 1)
]

# Entirety of Pacific Spirit Park!
pacific_spirit = [
    ( 1, 2, 1), ( 2, 3, 1), ( 3, 4, 1), ( 3,25, 6), ( 2, 4, 1), ( 2,24, 7),
    ( 4, 5, 3), ( 5, 6, 1), ( 5,26, 6), ( 6, 7, 1), ( 6, 8, 1), ( 8, 9,12),
    ( 8,26, 7), ( 9,10, 5), ( 9,19, 4), (10,11, 1), (10,19, 1), (10,12, 2),
    (12,13, 4), (12,18, 1), (13,14, 1), (13,15, 1), (13,16, 4), (16,31, 1),
    (31,30, 2), (30,94, 2), (94,17, 1), (17,18, 8), (18,19, 1), (18,20, 1),
    (19,20, 1), (20,21, 5), (21,22, 1), (21,25, 4), (22,23, 1), (22,24, 5),
    (24,25, 3), (25,26, 1), (26,27, 1), (28,103,3), (28,36, 4), (28,40, 1),
    (16,32, 1), (32,33, 1), (32,31, 1), (32,35, 3), (31,34, 2), (34,30, 1),
    (34,36, 1), (30,29, 3), (29,28, 1), (29,103,5), (29,95, 1), (17,95, 3),
    (94,95, 3), (96,103,2), (96,100,1), (96,101,1), (97,100,1), (98,100,1),
    (98,99, 1), (98,101,1), (101,102,1), (95,96,6), (41,103,1), (36,39, 1),
    (35,36, 1), (35,38, 1), (38,37, 1), (38,39, 1), (39,40, 4), (39,59, 2),
    (40,58, 3), (40,41, 3), (41,42, 3),
    (41,58, 2), (42,43, 1), (43,44, 2), (44,45, 1), (44,46, 1), (43,46, 2),
    (45,46, 1), (46,47, 1), (45,47, 1), (47,48, 3), (48,49, 3), (48,55, 1),
    (45,49, 1),
    (49,50, 4), (49,51,10), (51,52, 1), (51,90, 6), (51,91, 2), (48,52, 6),
    (52,53, 4), (53,54, 1), (53,92, 3), (54,55, 1), (54,60, 4), (55,56, 3),
    (42,56, 5), (56,57, 1), (57,58, 3), (57,60, 2), (60,61, 2), (61,62, 1),
    (61,65, 2), (62,59, 6), (62,63, 2), (63,64, 1), (63,65, 2), (64,68, 2),
    (64,70, 2), (59,70, 7), (70,71, 1), (71,72, 1), (38,71, 7), (71,74, 3),
    (74,73, 1), (74,69, 1), (69,70, 2), (68,69, 1), (67,68, 3), (66,67, 1),
    (66,78, 5), (65,66, 3), (53,65, 4), (66,93, 2), (67,77, 3), (68,77, 1),
    (69,75, 1), (75,76, 2), (75,77, 1), (75,79, 2), (77,78, 2), (78,79, 1),
    (79,80, 2), (79,81, 6), (81,82, 1), (81,83,16), (83,84, 5), (83,86, 1),
    (84,85, 1), (85,86, 4), (85,87, 6), (86,87, 6), (86,93,14), (87,88, 1),
    (87,89, 4), (87,90, 4), (90,91, 6), (91,92, 1), (92,93, 1)
]
