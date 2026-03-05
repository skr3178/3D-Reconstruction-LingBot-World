# Here are the point cloud generated on the image setup. 

The original is from the lingbot world model reconstruction while the generated is using the vggt based package for the generation. 

![orig](<Screenshot 2026-03-05 at 5.38.33 PM.png>)

![generated](<Screenshot 2026-03-05 at 5.38.41 PM.png>)

This repo contains the both point clouds and the scripts used to generate them. 

Original: /media/skr/storage/lingbot/3d_reconstruction_videos/scene2_pointcloud.glb

Generated: /media/skr/storage/lingbot/3d_reconstruction_videos/scene2_pointcloud_final.glb

## Commands used to generate scene2_pointcloud_final.glb

### Step 1 — Generate base point cloud from video using VGGT

```bash
cd /media/skr/storage/lingbot/3d_reconstruction_videos
/media/skr/storage/conda_envs/lingbot/bin/python vggt_video_to_glb.py \
  --video <input_video.mp4> \
  --output scene2_pointcloud.glb \
  --checkpoint VGGT-1B-Commercial/vggt_1B_commercial.pt \
  --max_frames 20
```

### Step 2 — Apply coordinate transforms (Y flip, Z flip, then 180° Z rotation)

```bash
/media/skr/storage/conda_envs/lingbot/bin/python - <<'EOF'
import trimesh, numpy as np

# Flip Y and Z axes
m = trimesh.load('scene2_pointcloud.glb')
geo = list(m.geometry.values())[0] if isinstance(m, trimesh.Scene) else m
geo.vertices[:, 1] *= -1
geo.vertices[:, 2] *= -1
geo.export('scene2_pointcloud_vggt_20f_clean_zflip.glb')

# Apply 180 degree rotation around Z axis (negate X and Y)
m2 = trimesh.load('scene2_pointcloud_vggt_20f_clean_zflip.glb')
geo2 = list(m2.geometry.values())[0] if isinstance(m2, trimesh.Scene) else m2
geo2.vertices[:, 0] *= -1
geo2.vertices[:, 1] *= -1
geo2.export('scene2_pointcloud_final.glb')
print('Done')
EOF
```


