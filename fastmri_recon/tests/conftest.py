import os

import h5py
import numpy as np
import pytest
import tensorflow as tf
from tqdm import tqdm

K_shape_single_coil = (2, 640, 322)
K_shape_multi_coil = (2, 15, 640, 322)
I_shape = (2, 320, 320)
contrast = 'CORPD_FBK'
fake_xml = b"""<?xml version="1.0" encoding="utf-8"?>\n<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.ismrm.org/ISMRMRD ismrmrd.xsd">\n   <studyInformation>\n      <studyTime>16:35:20</studyTime>\n   </studyInformation>\n   <measurementInformation>\n      <measurementID>41964_51683669_51683678_4583</measurementID>\n      <patientPosition>HFS</patientPosition>\n
<protocolName>AX FLAIR_FBB</protocolName>\n      <frameOfReferenceUID>1.3.12.2.1107.5.2.18.41964.1.20190707162548138.0.0.5022</frameOfReferenceUID>\n   </measurementInformation>\n   <acquisitionSystemInformation>\n      <systemVendor>SIEMENS</systemVendor>\n      <systemModel>Aera</systemModel>\n      <systemFieldStrength_T>1.494</systemFieldStrength_T>\n      <relativeReceiverNoiseBandwidth>0.793</relativeReceiverNoiseBandwidth>\n      <receiverChannels>16</receiverChannels>\n\
<coilLabel>\n         <coilNumber>29</coilNumber>\n         <coilName>HeadNeck_20:1:H13</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>30</coilNumber>\n         <coilName>HeadNeck_20:1:H14</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>17</coilNumber>\n         <coilName>HeadNeck_20:1:H23</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>18</coilNumber>\n         <coilName>HeadNeck_20:1:H24</coilName>\n\
</coilLabel>\n      <coilLabel>\n         <coilNumber>45</coilNumber>\n         <coilName>HeadNeck_20:1:H32</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>46</coilNumber>\n         <coilName>HeadNeck_20:1:H31</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>21</coilNumber>\n         <coilName>HeadNeck_20:1:H11</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>22</coilNumber>\n\
<coilName>HeadNeck_20:1:H12</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>34</coilNumber>\n         <coilName>HeadNeck_20:1:H42</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>33</coilNumber>\n         <coilName>HeadNeck_20:1:H41</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>38</coilNumber>\n         <coilName>HeadNeck_20:1:H44</coilName>\n      </coilLabel>\n      <coilLabel>\n\
<coilNumber>37</coilNumber>\n         <coilName>HeadNeck_20:1:H43</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>26</coilNumber>\n         <coilName>HeadNeck_20:1:H34</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>25</coilNumber>\n         <coilName>HeadNeck_20:1:H33</coilName>\n      </coilLabel>\n      <coilLabel>\n         <coilNumber>42</coilNumber>\n         <coilName>HeadNeck_20:1:H21</coilName>\n      </coilLabel>\n\
<coilLabel>\n         <coilNumber>41</coilNumber>\n         <coilName>HeadNeck_20:1:H22</coilName>\n      </coilLabel>\n      <institutionName>TH RADIOLOGY</institutionName>\n   </acquisitionSystemInformation>\n   <experimentalConditions>\n      <H1resonanceFrequency_Hz>63690796</H1resonanceFrequency_Hz>\n   </experimentalConditions>\n   <encoding>\n      <encodedSpace>\n         <matrixSize>\n            <x>640</x>\n            <y>264</y>\n            <z>1</z>\n         </matrixSize>\n\
<fieldOfView_mm>\n            <x>440</x>\n            <y>181.4312</y>\n            <z>7.5</z>\n         </fieldOfView_mm>\n      </encodedSpace>\n      <reconSpace>\n         <matrixSize>\n            <x>320</x>\n            <y>260</y>\n            <z>1</z>\n         </matrixSize>\n         <fieldOfView_mm>\n            <x>220</x>\n            <y>178.75</y>\n            <z>5</z>\n         </fieldOfView_mm>\n      </reconSpace>\n      <trajectory>cartesian</trajectory>\n\
<encodingLimits>\n         <kspace_encoding_step_1>\n            <minimum>0</minimum>\n            <maximum>197</maximum>\n            <center>99</center>\n         </kspace_encoding_step_1>\n         <kspace_encoding_step_2>\n            <minimum>0</minimum>\n            <maximum>0</maximum>\n            <center>0</center>\n         </kspace_encoding_step_2>\n         <average>\n            <minimum>0</minimum>\n            <maximum>0</maximum>\n            <center>0</center>\n\
</average>\n         <slice>\n            <minimum>0</minimum>\n            <maximum>33</maximum>\n            <center>0</center>\n         </slice>\n         <contrast>\n            <minimum>0</minimum>\n            <maximum>0</maximum>\n            <center>0</center>\n         </contrast>\n         <phase>\n            <minimum>0</minimum>\n            <maximum>0</maximum>\n            <center>0</center>\n         </phase>\n         <repetition>\n            <minimum>0</minimum>\n\
<maximum>0</maximum>\n            <center>0</center>\n         </repetition>\n         <set>\n            <minimum>0</minimum>\n            <maximum>0</maximum>\n            <center>0</center>\n         </set>\n         <segment>\n            <minimum>0</minimum>\n            <maximum>0</maximum>\n            <center>0</center>\n         </segment>\n      </encodingLimits>\n      <parallelImaging>\n         <accelerationFactor>\n\
<kspace_encoding_step_1>1</kspace_encoding_step_1>\n            <kspace_encoding_step_2>1</kspace_encoding_step_2>\n         </accelerationFactor>\n         <calibrationMode>other</calibrationMode>\n      </parallelImaging>\n   </encoding>\n   <sequenceParameters>\n      <TR>9000</TR>\n      <TE>86</TE>\n      <TI>2500</TI>\n      <flipAngle_deg>150</flipAngle_deg>\n      <sequence_type>TurboSpinEcho</sequence_type>\n      <echo_spacing>9.6</echo_spacing>\n   </sequenceParameters>\n\
<userParameters>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_0</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_1</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_2</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_3</name>\n\
<value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_4</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_5</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_6</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n\
<name>MaxwellCoefficient_7</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_8</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_9</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_10</name>\n         <value>0</value>\n      </userParameterDouble>\n\
<userParameterDouble>\n         <name>MaxwellCoefficient_11</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_12</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_13</name>\n         <value>0</value>\n      </userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_14</name>\n         <value>0</value>\n\
</userParameterDouble>\n      <userParameterDouble>\n         <name>MaxwellCoefficient_15</name>\n         <value>0</value>\n      </userParameterDouble>\n   </userParameters>\n</ismrmrdHeader>"""

CI = os.environ.get('CONTINUOUS_INTEGRATION', False) == 'true'
CI = CI or os.environ.get('CI', False) == 'true'
CI = CI or os.environ.get('TRAVIS', False) == 'true'

os.environ['CI'] = str(CI)

def create_data(filename, multicoil=False, train=True):
    k_shape = K_shape_single_coil
    if multicoil:
        k_shape = K_shape_multi_coil
    if train:
        image_ds = "reconstruction_esc"
        if multicoil:
            image_ds = "reconstruction_rss"
        image = np.random.normal(size=I_shape)
        image = image.astype(np.float32)
    else:
        mask_shape = [K_shape_multi_coil[-1]]
        mask = np.random.choice(a=[True, False], size=mask_shape)
        af = np.sum(mask.astype(int)) / mask_shape[0]
    kspace = np.random.normal(size=k_shape) + 1j * np.random.normal(size=k_shape)
    kspace = kspace.astype(np.complex64)
    with h5py.File(filename, "w") as h5_obj:
        h5_obj.create_dataset("kspace", data=kspace)
        if train:
            h5_obj.create_dataset(image_ds, data=image)
        else:
            h5_obj.create_dataset('mask', data=mask)
            h5_obj.create_dataset('ismrmrd_header', data=fake_xml)
        h5_obj.attrs['acquisition'] = contrast
    if not train:
        return af

@pytest.fixture(scope='session', autouse=False)
def ktraj():
    def ktraj_function(image_shape, nspokes):
        # radial trajectory creation
        spokelength = image_shape[-1] * 2
        nspokes = 15

        ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
        kx = np.zeros(shape=(spokelength, nspokes))
        ky = np.zeros(shape=(spokelength, nspokes))
        ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
        for i in range(1, nspokes):
            kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
            ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

        ky = np.transpose(ky)
        kx = np.transpose(kx)

        traj = np.stack((ky.flatten(), kx.flatten()), axis=0)
        traj = tf.convert_to_tensor(traj)[None, ...]
        return traj
    return ktraj_function


@pytest.fixture(scope="session", autouse=False)
def create_full_fastmri_test_tmp_dataset(tmpdir_factory):
    # main dirs
    fastmri_tmp_data_dir = tmpdir_factory.mktemp(
        "fastmri_test_tmp_data",
        numbered=False,
    )
    logs_tmp_dir = tmpdir_factory.mktemp(
        "logs",
        numbered=False,
    )
    checkpoints_tmp_dir = tmpdir_factory.mktemp(
        "checkpoints",
        numbered=False,
    )
    #### single coil
    fastmri_tmp_singlecoil_train = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('singlecoil_train')
    ), numbered=False)
    fastmri_tmp_singlecoil_train = tmpdir_factory.mktemp(str(
        fastmri_tmp_singlecoil_train.join('singlecoil_train')
    ), numbered=False)
    fastmri_tmp_singlecoil_val = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('singlecoil_val')
    ), numbered=False)
    fastmri_tmp_singlecoil_test = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('singlecoil_test')
    ), numbered=False)
    n_files = 2
    # train
    for i in tqdm(range(n_files), 'Creating single coil train files'):
        data_filename = f"train_singlecoil_{i}.h5"
        create_data(str(fastmri_tmp_singlecoil_train.join(data_filename)))
    # val
    for i in tqdm(range(n_files), 'Creating single coil val files'):
        data_filename = f"val_singlecoil_{i}.h5"
        create_data(str(fastmri_tmp_singlecoil_val.join(data_filename)))
    # test
    af_single_coil = []
    for i in tqdm(range(n_files), 'Creating single coil test files'):
        data_filename = f"test_singlecoil_{i}.h5"
        af = create_data(
            str(fastmri_tmp_singlecoil_test.join(data_filename)),
            multicoil=False,
            train=False,
        )
        af_single_coil.append(af)
    #### multi coil
    fastmri_tmp_multicoil_train = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('multicoil_train')
    ), numbered=False)
    fastmri_tmp_multicoil_val = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('multicoil_val')
    ), numbered=False)
    fastmri_tmp_multicoil_test = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('multicoil_test')
    ), numbered=False)
    n_files = 2
    # train
    for i in tqdm(range(n_files), 'Creating multi coil train files'):
        data_filename = f"train_multicoil_{i}.h5"
        create_data(
            str(fastmri_tmp_multicoil_train.join(data_filename)),
            multicoil=True,
        )
    # val
    for i in tqdm(range(n_files), 'Creating multi coil val files'):
        data_filename = f"val_multicoil_{i}.h5"
        create_data(
            str(fastmri_tmp_multicoil_val.join(data_filename)),
            multicoil=True,
        )
    # test
    af_multi_coil = []
    for i in tqdm(range(n_files), 'Creating multi coil test files'):
        data_filename = f"test_multicoil_{i}.h5"
        af = create_data(
            str(fastmri_tmp_multicoil_test.join(data_filename)),
            multicoil=True,
            train=False,
        )
        af_multi_coil.append(af)

    return {
        'fastmri_tmp_data_dir': str(fastmri_tmp_data_dir) + '/',
        'logs_tmp_dir': str(tmpdir_factory.getbasetemp()) + '/',
        'checkpoints_tmp_dir': str(tmpdir_factory.getbasetemp()) + '/',
        'fastmri_tmp_singlecoil_train': str(fastmri_tmp_singlecoil_train) + '/',
        'fastmri_tmp_singlecoil_val': str(fastmri_tmp_singlecoil_val) + '/',
        'fastmri_tmp_singlecoil_test': str(fastmri_tmp_singlecoil_test) + '/',
        'fastmri_tmp_multicoil_train': str(fastmri_tmp_multicoil_train) + '/',
        'fastmri_tmp_multicoil_val': str(fastmri_tmp_multicoil_val) + '/',
        'fastmri_tmp_multicoil_test': str(fastmri_tmp_multicoil_test) + '/',
        'af_single_coil': af_single_coil,
        'af_multi_coil': af_multi_coil,
        'K_shape_single_coil': K_shape_single_coil,
        'K_shape_multi_coil': K_shape_multi_coil,
        'I_shape': I_shape,
        'contrast': contrast,
    }
