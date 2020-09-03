import subprocess


GRAPPA_RECON_DS_NAME = 'recon_grappa_generic'

def gadgetron_execution(in_file, out_file, config_name, out_group):
    subprocess.check_call([
        'gadgetron_ismrmrd_client',
        '-f', str(in_file),
        '-o', str(out_file),
        '-c', config_name,
        '-G', out_group
    ])

def gadgetron_grappa_reconstruction(in_file, out_file, my_config=False):
    config = 'Generic_Cartesian_Grappa.xml'
    if my_config:
        config = 'my_grappa_gadgetron_config.xml'
    gadgetron_execution(
        in_file,
        out_file,
        config,
        GRAPPA_RECON_DS_NAME,
    )
