#!/usr/bin/env bash

source .env

if [[ -z "$1" ]]; then
  echo "Name of backup file (located in "$DB_BACKUP_LOCAL") required!"
  exit 1
fi

DB_CONTAINER_ID=$(docker-compose ps -q db)

if [ -z "$DB_CONTAINER_ID" ] || [ -z "$(docker ps -q --no-trunc | grep $DB_CONTAINER_ID)" ]; then
  echo "Database container is not running!"
  exit 1
fi

RESTORE_COMMANDS="set "PGPASSWORD=$DB_PASSWORD" && pg_restore -v -c -d "$DB_NAME" "$DB_BACKUP_DOCKER""$1".tar -U "$DB_USER""

docker exec "$DB_CONTAINER_ID" bash -c "$RESTORE_COMMANDS"