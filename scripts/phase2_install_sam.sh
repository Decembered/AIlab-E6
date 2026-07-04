#!/bin/bash
set -e
echo "Installing SAM dependencies..."
pip install segment-anything opencv-python matplotlib -q
echo "Done."
python -c "import segment_anything; print('SAM imported OK')"
