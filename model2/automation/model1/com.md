docker_command = f"""
        docker run --rm -i --name snowflake --entrypoint sh  -c "
        ansible-playbook -i /ansible/ansible/inventory /ansible/ansible/auto.yml --vault-password-file /ansible/ansible/vars/vp.txt
        "
    """
        docker run --rm -i --name {github_project['repo_name']} --entrypoint sh lubnaibrahimu/obelion-ansible:latest -c "
        sed -i 's/^frontend_domain:./frontend_domain: \"{frontend_domain}\"/' /ansible/ansible/vars/all.yml &&
        sed -i 's/^backend_domain:./backend_domain: \"{backend_domain}\"/' /ansible/ansible/vars/all.yml &&
        sed -i 's|^app_repo:.|app_repo: \"{repo_url}\"|' /ansible/ansible/vars/all.yml &&
        sed -i 's/^ec2_name:./ec2_name: \"{ec2_name}\"/' /ansible/ansible/vars/all.yml &&
        sed -i 's|^appDir:.|appDir: {app_dir}|' /ansible/ansible/vars/all.yml &&
        sed -i 's/^DB_NAME:./DB_NAME: \"{db_name}\"/' /ansible/ansible/vars/all.yml &&
        ansible-playbook -i /ansible/ansible/inventory /ansible/ansible/auto.yml --vault-password-file /ansible/ansible/vars/vp.txt
        "


docker run --rm -i --name snowflake --entrypoint sh lubnaibrahimu/snow-chromadb:latest -c "
sed -i 's|^appDir:.*|appDir: chromadb|' /ansible/ansible/vars/all.yml
sed -i 's|^container_name:.*|container_name: chromadb1|' /ansible/ansible/vars/all.yml
sed -i 's|^snow_domain:.*|snow_domain: snowchromadb|' /ansible/ansible/vars/all.yml
ansible-playbook -i /ansible/ansible/inventory /ansible/ansible/snow.yml  --vault-password-file /ansible/ansible/vars/vp.txt
"