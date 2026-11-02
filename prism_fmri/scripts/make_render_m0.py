# render_eval.py(f3b919ad) 사본에 GLB 재질 metallic 0 고정만 넣어 render_eval_m0.py로 저장
#  이유: TRELLIS GLB는 metallicFactor가 없어 glTF 기본값 1.0(금속)으로 들어가 어둡게 렌더됨 (10-30 확인)
#  공용 render_eval.py는 기준선도 쓰므로 그대로 둔다. OBJ 경로는 바뀌지 않는다.
# 사용: python make_render_m0.py <render_eval.py> <render_eval_m0.py>
import sys, hashlib
src, dst = sys.argv[1:3]
b = open(src, 'rb').read()
h = hashlib.md5(b).hexdigest()
assert h.startswith('f3b919ad'), f'render_eval.py md5 {h} != f3b919ad'
s = b.decode('utf-8')
old = '''        bpy.ops.import_scene.gltf(filepath=p, merge_vertices=True)
        bpy.context.view_layer.update()
        return'''
new = '''        bpy.ops.import_scene.gltf(filepath=p, merge_vertices=True)
        bpy.context.view_layer.update()
        for m in bpy.data.materials:   # metallic 0 고정 (TRELLIS GLB의 glTF 기본값 1.0 대신)
            n = m.node_tree.nodes.get('Principled BSDF') if m.use_nodes else None
            if n: n.inputs['Metallic'].default_value = 0.0
        return'''
assert s.count(old) == 1
open(dst, 'w', encoding='utf-8').write(s.replace(old, new))
print('wrote', dst)
