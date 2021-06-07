import os, tempfile, shutil

# Generate Q# build config by running the SDK
# Input: the source file directory where the build
# csproj will be saved to.
def generate_qsc_build_config(src_dir)
  temp_dir = tempfile.mkdtemp()
  # print(temp_dir)
  os.system('dotnet new console -lang Q# -o ' + temp_dir + ' -n test')

  import xml.etree.ElementTree as ET
  # Parse the SDK generated Project file
  tree = ET.parse(temp_dir + '/test.csproj')

  # get root element
  root = tree.getroot()
  print(root.tag, root.attrib, root.text)
  # run this target:
  root.attrib['DefaultTargets'] = 'PrepareQSharpCompile'
  propertyGroup = root.find('PropertyGroup')
  # enable QIR generation
  qir_gen = ET.Element('QirGeneration')
  qir_gen.text = 'true'
  propertyGroup.append(qir_gen)

  # Add a target to query the actual Build Command
  # to get the most accurate QSC executable:
  # <Target Name="PrintBuildCommand" BeforeTargets="PrepareQSharpCompile" >
  #   <WriteLinesToFile File="QscExe.cmd" Overwrite="true" Lines="$(QscExe)" />
  # </Target>

  export_qsc_cmd = ET.Element('WriteLinesToFile')
  export_qsc_cmd.attrib['File'] = 'QscExe.cmd'
  export_qsc_cmd.attrib['Overwrite'] = 'true'
  export_qsc_cmd.attrib['Lines'] = '$(QscExe)'

  export_qsc_cmd_target = ET.Element('Target')
  export_qsc_cmd_target.attrib['Name'] = 'PrintBuildCommand'
  export_qsc_cmd_target.attrib['BeforeTargets'] = 'PrepareQSharpCompile'
  export_qsc_cmd_target.append(export_qsc_cmd)

  root.append(export_qsc_cmd_target)
  tree.write(src_dir + '/output.csproj')
  # Delete the temp_directory:
  shutil.rmtree(temp_dir)
