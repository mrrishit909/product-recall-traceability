-- Domain model. Names follow the blueprint's schema; additions are marked (+). Times are timestamptz; the synthetic plant's
-- production starts 2026-07-01.

CREATE TABLE plant (plant_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, seed int NOT NULL, as_of timestamptz NOT NULL, metadata jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (tenant_id));   -- (+) one plant per tenant here
SELECT enable_tenant_rls('plant');

-- A supplier lot as its advance ship notice describes it: reels per date code. The notice is the reference the reels are matched to.
CREATE TABLE material_batch (material_batch_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, external_ref text, name text, metadata jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (tenant_id, external_ref));
SELECT enable_tenant_rls('material_batch');

-- A reel as receiving recorded it. material_batch_id stays NULL: which lot a reel came from is inferred (genealogy_edge), because the
-- read code can be wrong or missing.
CREATE TABLE component_lot (component_lot_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, material_batch_id uuid REFERENCES material_batch, status text NOT NULL, attributes jsonb NOT NULL DEFAULT '{}', observed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(),
  external_ref text NOT NULL, delivery_ref text NOT NULL, lot_code_read text, date_code_read date, UNIQUE (tenant_id, external_ref));   -- (+) reel id, delivery, what the scanner read
SELECT enable_tenant_rls('component_lot');

CREATE TABLE serial_unit (unit_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, serial text NOT NULL, kind text NOT NULL CHECK (kind IN ('drive_unit', 'controller_board')), line smallint, built_at timestamptz NOT NULL, work_order text,
  board_serial text, spare boolean NOT NULL DEFAULT false, UNIQUE (tenant_id, serial));     -- board_serial: the board fitted when a drive unit was built
SELECT enable_tenant_rls('serial_unit');
CREATE INDEX serial_unit_board ON serial_unit (tenant_id, board_serial);

-- Shop-floor events as the systems logged them, flaws included. event_id is the source system's id: replays are ignored.
CREATE TABLE assembly_event (tenant_id uuid NOT NULL REFERENCES tenants, event_id text NOT NULL, kind text NOT NULL CHECK (kind IN ('reel_issue', 'reel_scan', 'splice_sensor', 'rework')), at timestamptz NOT NULL, line smallint, reel_ref text, serial text, board_serial text, detail jsonb NOT NULL DEFAULT '{}', recorded_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (tenant_id, event_id));
SELECT enable_tenant_rls('assembly_event');
CREATE INDEX assembly_event_serial ON assembly_event (tenant_id, serial);

CREATE TABLE shipment (shipment_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, external_ref text NOT NULL, customer text NOT NULL, shipped_at timestamptz NOT NULL, UNIQUE (tenant_id, external_ref));
SELECT enable_tenant_rls('shipment');

CREATE TABLE customer_delivery (unit_id uuid PRIMARY KEY REFERENCES serial_unit, shipment_id uuid NOT NULL REFERENCES shipment, tenant_id uuid NOT NULL REFERENCES tenants);
SELECT enable_tenant_rls('customer_delivery');
CREATE INDEX customer_delivery_shipment ON customer_delivery (shipment_id);

CREATE TABLE quality_event (quality_event_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, kind text NOT NULL CHECK (kind IN ('field_return', 'supplier_notice')), serial text, at timestamptz NOT NULL, detail jsonb NOT NULL DEFAULT '{}', created_by uuid);
SELECT enable_tenant_rls('quality_event');

-- The inferred genealogy: reel -> lot, reel -> board, and board -> unit where a rework ticket lost the serial, each with its probability and the model version that produced it.
CREATE TABLE genealogy_edge (tenant_id uuid NOT NULL REFERENCES tenants, model_version text NOT NULL, relation text NOT NULL CHECK (relation IN ('reel_of_lot', 'reel_in_board', 'board_in_unit')), parent_ref text NOT NULL, child_ref text NOT NULL, probability real NOT NULL CHECK (probability > 0 AND probability <= 1.0001), PRIMARY KEY (tenant_id, model_version, relation, parent_ref, child_ref));
SELECT enable_tenant_rls('genealogy_edge');
CREATE INDEX genealogy_edge_child ON genealogy_edge (tenant_id, model_version, child_ref);

CREATE TABLE model_artifact (model_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, model_name text NOT NULL, version text NOT NULL, artifact_uri text NOT NULL, feature_schema jsonb NOT NULL, metrics jsonb NOT NULL DEFAULT '{}', approved boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now(), params jsonb NOT NULL, UNIQUE (tenant_id, model_name, version));   -- (+) params
SELECT enable_tenant_rls('model_artifact');

CREATE TABLE recall_case (recall_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, incident_id uuid NOT NULL REFERENCES quality_event, status text NOT NULL CHECK (status IN ('scoping', 'proposed', 'approved')), params jsonb NOT NULL, metrics jsonb NOT NULL DEFAULT '{}', created_by uuid NOT NULL, approved_by uuid, approved_at timestamptz, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('recall_case');

-- Every unit any strategy would recall, with the model's probability and the cost to get it back.
CREATE TABLE recall_scope (recall_id uuid NOT NULL REFERENCES recall_case, unit_id uuid NOT NULL REFERENCES serial_unit, tenant_id uuid NOT NULL REFERENCES tenants, serial text NOT NULL, p_affected real NOT NULL, cost real NOT NULL, customer text, shipment_ref text, stage text NOT NULL,
  in_model boolean NOT NULL, in_mes boolean NOT NULL, in_work_order boolean NOT NULL, in_broad boolean NOT NULL, PRIMARY KEY (recall_id, unit_id));
SELECT enable_tenant_rls('recall_scope');

CREATE TABLE decision_record (decision_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, subject_type text NOT NULL, subject_id uuid NOT NULL, decision_type text NOT NULL, model_id uuid REFERENCES model_artifact, inputs_hash text NOT NULL, result jsonb NOT NULL, confidence double precision, policy_state jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('decision_record');
