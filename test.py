import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
from skimage.transform import resize


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.nThreads = 1
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        if opt.phase == 'testcube':
            # 3D from 2D patches
            print('3D from 2D patches/slices...')

            N_slices = data['A'].shape[2] 
            cubes = dict([(x, []) for x in model.visual_names])
            for slice_i in range(N_slices):
                print(slice_i, N_slices)
                slice_data = dict()
                slice_data['A'] = data['A'][:,:,slice_i]
                slice_data['B'] = data['B'][:,:,slice_i]
                slice_data['A_paths'] = data['A_paths']
                slice_data['B_paths'] = data['B_paths']
                model.set_input(slice_data)
                model.test()
                for visual_name in model.visual_names:
                    cubes[visual_name].append(getattr(model, visual_name)[0])
            for visual_name in model.visual_names:
                path = os.path.join(web_dir, '%s_%s_%s' % (i, opt.name.replace('/', '_'), visual_name))
                data_cube = np.concatenate(cubes[visual_name])
                np.save(path, data_cube)
        elif opt.phase == 'testcube3d':
            if opt.test_window == -1:
                # 3D from 3D volume
                print('3D from 3D volume...')
                model.set_input(data)
                model.test()

                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()
                print('processing (%04d)-th image... %s' % (i, img_path))

                if opt.no_output_tanh:
                    scale_range = (-1, 0.2)
                else:
                    scale_range = (-1, 1)
                save_images(webpage, visuals, img_path, scale_range, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
                for visual_name in model.visual_names:
                    path = os.path.join(web_dir, '%s_%s_%s' % (i, opt.name.replace('/', '_'), visual_name))
                    data_cube = getattr(model, visual_name)[0]
                    np.save(path, data_cube)
            else:
                # 3D from 3D patches
                print('3D from 3D patches...')
                x_len, y_len, z_len = data['A'].shape[2:]
                print('A:', data['A'].shape)
                print('B:', data['B'].shape)
                print('len(x), len(y), len(z)=', (x_len, y_len, z_len))
                # p = opt.test_window
                
                # Crazy moving window code
                # inner cube has dimensions = (px, py, pz)
                # outer cube has dimensions = (2px, 2py, 2pz)
                # inner cube walks through volume, outer cube moves accordingly

                px, py, pz = opt.test_window, opt.test_window, opt.test_window
                x_cubes = dict([(x, []) for x in model.visual_names])
                for x in range(0, x_len, px):
                    y_cubes = dict([(x, []) for x in model.visual_names])
                    for y in range(0, y_len, py):
                        print('x, y, z:', x, y, '~')
                        z_cubes = dict([(x, []) for x in model.visual_names])
                        for z in range(0, z_len, pz):
                            print('x, y, z:', x, y, z)
                            xl = -min(0, x - (px//4))
                            xr = max(x_len, x + 3*(px//4)) - x_len
                            yl = -min(0, y - (py//4))
                            yr = max(y_len, y + 3*(py//4)) - y_len
                            zl = -min(0, z - (pz//4))
                            zr = max(z_len, z + 3*(pz//4)) - z_len
                            assert not (xl and xr)
                            assert not (yl and yr)
                            assert not (zl and zr)
                            bx1 = x - (px//4) - xr + xl
                            by1 = y - (py//4) - yr + yl
                            bz1 = z - (pz//4) - zr + zl
                            bx2 = min(x_len, bx1 + 2*px)
                            by2 = min(y_len, by1 + 2*py)
                            bz2 = min(z_len, bz1 + 2*pz)

                            patch_data = dict()
                            patch_data['A'] = data['A'][:, :, bx1:bx2, by1:by2, bz1:bz2]
                            patch_data['B'] = data['B'][:, :, bx1:bx2, by1:by2, bz1:bz2]
                            patch_data['A_paths'] = data['A_paths']
                            patch_data['B_paths'] = data['B_paths']

                            model.set_input(patch_data)
                            model.test()
                            inx = (px // 4) - xl + xr
                            iny = (py // 4) - yl + yr
                            inz = (pz // 4) - zl + zr
                            for visual_name in model.visual_names:
                                z_cubes[visual_name].append(getattr(model, visual_name)[0, 0, inx:inx+px, iny:iny+py, inz:inz+pz].detach().cpu().numpy())

                        for visual_name in model.visual_names:
                            y_cubes[visual_name].append(np.concatenate(z_cubes[visual_name], axis=2))
                    for visual_name in model.visual_names:
                        x_cubes[visual_name].append(np.concatenate(y_cubes[visual_name], axis=1))

                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()
                print('processing (%04d)-th image... %s' % (i, img_path))

                if opt.no_output_tanh:
                    scale_range = (-1, 0.2)
                else:
                    scale_range = (-1, 1)
                save_images(webpage, visuals, img_path, scale_range, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
                for visual_name in model.visual_names:
                    path = os.path.join(web_dir, '%s_%s_%s' % (i, opt.name.replace('/', '_'), visual_name))
                    data_cube = np.concatenate(x_cubes[visual_name], axis=0)
                    print('Saving:', data_cube.shape)
                    np.save(path, data_cube)
        else:
            # 2D from 2D
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))

            if opt.no_output_tanh:
                scale_range = (-1, 0.2)
            else:
                scale_range = (-1, 1)
            save_images(webpage, visuals, img_path, scale_range, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
