# Map format

Maps are deliberately plain text so experiments can version and hash them without a game editor.
The v1 format accepts `name`, `start`, `finish`, and repeated `point=x,y` records. Points must have
strictly increasing x coordinates. Terrain between points is linearly interpolated. The start and
finish must lie inside the terrain bounds.

All coordinates use meters and positive y points upward. Comments begin with `#`. A training result
should record both the map path and its file hash; changing a point creates a different experiment.
